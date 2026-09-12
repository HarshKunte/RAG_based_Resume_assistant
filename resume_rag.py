import re
import os
from pathlib import Path
from uuid import uuid4

import chromadb
import dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from fs_tools import list_files, read_file
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

dotenv.load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. "
        "Add it to your .env file."
    )

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

persist_directory = "./chroma_langchain_db"


SECTION_NAMES = {
    "SUMMARY",
    "PROFESSIONAL EXPERIENCE",
    "EXPERIENCE",
    "SKILLS",
    "PROJECTS",
    "CERTIFICATIONS",
    "EDUCATION",
}

persistent_client = chromadb.PersistentClient(path=persist_directory)
collection_name = "my_documents"

def get_vector_db():
    try:
        # Attempt to grab the collection directly
        collection = persistent_client.get_collection(name=collection_name)
        # print(f"Collection '{collection_name}' exists. Loading...")
        
        return Chroma(
            client=persistent_client,
            collection_name=collection_name,
            embedding_function=embeddings
        )
    except (ValueError, Exception):
        return read_file_data_and_create_vector_db()
    

def split_resume_sections(text: str) -> list[tuple[str, str]]:
    heading_pattern = "|".join(
        re.escape(section) for section in SECTION_NAMES
    )

    pattern = re.compile(
        rf"(?im)^\s*({heading_pattern})\s*$"
    )

    matches = list(pattern.finditer(text))
    sections = []

    if not matches:
        return [("unknown", text)]

    # Content before the first heading
    if matches[0].start() > 0:
        sections.append(("header", text[:matches[0].start()]))

    for index, match in enumerate(matches):
        section_name = match.group(1).strip().lower()

        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)

        section_text = text[start:end].strip()

        if section_text:
            sections.append((section_name, section_text))

    return sections

def read_file_data_and_create_vector_db():
    vector_store = Chroma(
                    collection_name=collection_name,
                    embedding_function=embeddings,
                    persist_directory= persist_directory,
    )
    # Step 1: List files in the "resumes" directory with .pdf, .docx, and .txt extensions
    directory = "resumes"
    extensions = [".pdf", ".docx", ".txt"]
    all_files = []
    for ext in extensions:
        files = list_files(directory, ext)
        if "error" not in files[0]:
            all_files.extend(files)

    for file_info in all_files:
        documents = read_file(file_info["filepath"])
        # print(documents)

        # Step 3: Create a vector database from the documents
        # (This is a placeholder; implement your vector database creation logic here)
        create_vector_database_from_documents(documents, vector_store)
    return vector_store

def create_vector_database_from_documents(documents, vector_store):

    file_name = documents["filename"]
    resume_text = "\n".join(page.page_content for page in documents["content"])
    resume_metadata = extract_resume_metadata(resume_text, file_name)
    db_documents = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    for page in documents["content"]:
        sections = split_resume_sections(page.page_content)

        for section_name, section_text in sections:
            chunks = splitter.create_documents([section_text])

            for chunk_index, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "source": file_name,
                    "candidate_id": resume_metadata["candidate_id"],
                    "candidate_name": resume_metadata["candidate_name"],
                    "skills": ", ".join(resume_metadata["skills"]),
                    "experience_years": resume_metadata["experience_years"],
                    "education": ", ".join(resume_metadata["education"]),
                    "section": section_name,
                    "chunk_index": chunk_index,
                })

                chunk.metadata["keywords"] = ", ".join(
                    extract_keywords_from_documents(chunk)
                )

                db_documents.append(chunk)

    uuids = [str(uuid4()) for _ in db_documents]
    
    vector_store.add_documents(documents=db_documents, ids=uuids)
    # print(f"Added {len(db_documents)} chunks for {file_name}")



def extract_keywords_from_documents(document):
    class DocumentMetadata(BaseModel):
        keywords: list[str] = Field(
            description="5-10 specific keywords, tech terms, or entities found in the text."
        )

    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        temperature=0,
        timeout=30,
        openai_api_key=API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )
    structured_llm = llm.with_structured_output(DocumentMetadata)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert at resume analytics and keywords extraction system. "
            "Extract all relevant skills and keywords from the text generically. "
            "Do not limit extraction to a predefined vocabulary. "
        ),
        ("user", "{text}"),
    ])

    extraction = (prompt | structured_llm).invoke({"text": document.page_content})
    return extraction.keywords


def extract_requirements_from_job_description(job_description):
    class DocumentMetadata(BaseModel):
        must_have_skills: list[str] = Field(
            description="List of must have skills, tech terms, or entities found in the text."
        )
        nice_to_have_skills: list[str] = Field(
            description="List of nice to have skills, tech terms, or entities found in the text."
        )

    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        temperature=0,
        timeout=30,
        openai_api_key=API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )
    structured_llm = llm.with_structured_output(DocumentMetadata)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert at analyzing the skills required for a job and extracting them from the given job description."
            "Extract only technical and other relevant skills, divide them into must have and nice to have skills. Split skills by spaces also if required"
            "Do not limit extraction to a predefined vocabulary. "
            "Use only skills explicitly stated in the supplied job description. "
            "Never infer, add, or hallucinate a skill that is not present in the text. "
            "Give all upper, lower and other versions of the same skill as separate entries."
        ),
        ("user", "{text}"),
    ])

    extraction = (prompt | structured_llm).invoke({"text": job_description})
    # print(f"Extracted must have skills: {extraction.must_have_skills}")
    # print(f"Extracted nice to have skills: {extraction.nice_to_have_skills}")
    return {
        "must_have":extraction.must_have_skills,
        "nice_to_have":extraction.nice_to_have_skills
    }


def extract_resume_metadata(text: str, filename: str) -> dict:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        # Resume headers often contain email, LinkedIn, and GitHub details.
        # Keep only the person's name for clear candidate display.
        candidate_name = re.split(r"\s*(?:\||,|\bemail\b)", lines[0], maxsplit=1, flags=re.IGNORECASE)[0]
        candidate_name = re.sub(r"\s+", " ", candidate_name).strip()
    else:
        candidate_name = Path(filename).stem.replace("_", " ")

    years_match = re.search(r"(\d+(?:\.\d+)?)\+?\s+years?", text, re.IGNORECASE)
    experience_years = float(years_match.group(1)) if years_match else 0.0

    sections = split_resume_sections(text)
    skills_text = next((content for section, content in sections if section == "skills"), text)
    education_text = next((content for section, content in sections if section == "education"), "")
    skills = extract_keywords_from_documents(
        type("TextDocument", (), {"page_content": skills_text})()
    )
    education = [
        line[:200] for line in education_text.splitlines()
        if line.strip() and not re.fullmatch(r"[A-Z0-9\s–/-]+", line.strip())
    ][:5]

    return {
        "candidate_id": re.sub(
            r"[^a-z0-9]+", "_", Path(filename).stem.lower()
        ).strip("_"),
        "candidate_name": candidate_name,
        "skills": skills,
        "experience_years": experience_years,
        "education": education,
    }


if __name__ == "__main__":
    vector_store = get_vector_db()
    results = vector_store.similarity_search("What frontend experience does Harsh have?",
    k=5,
    filter={"section": "professional experience"})
    # print(results)