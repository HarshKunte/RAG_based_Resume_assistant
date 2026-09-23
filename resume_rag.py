import re
import os
from pathlib import Path
from uuid import uuid4

import chromadb
import dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from mcp_client import MCPFilesystemClient

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
    print("[DEBUG] get_vector_db: attempting to load or refresh Chroma collection")
    try:
        collection = persistent_client.get_collection(name=collection_name)
        print(f"[DEBUG] get_vector_db: collection found: {collection_name}")
        vector_store = Chroma(
            client=persistent_client,
            collection_name=collection_name,
            embedding_function=embeddings
        )
        print("[DEBUG] get_vector_db: calling index_new_resumes")
        index_new_resumes(vector_store)
        return vector_store
    except (ValueError, Exception) as exc:
        print(f"[DEBUG] get_vector_db: collection load failed, rebuilding DB: {exc}")
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

def _resume_paths(directory: str, client: MCPFilesystemClient) -> list[str]:
    paths = []
    for extension in (".pdf", ".docx", ".txt"):
        paths.extend(
            item["filepath"]
            for item in client.list_files(directory, extension)
            if "filepath" in item
        )
    return paths


def _indexed_resume_paths(vector_store, directory: str) -> set[str]:
    collection = getattr(vector_store, "_collection", None)
    if collection is None:
        return set()
    metadata_rows = collection.get(include=["metadatas"]).get("metadatas", [])
    return {
        str(Path(directory, metadata["source"]).resolve())
        for metadata in metadata_rows
        if metadata and metadata.get("source")
    }


def index_new_resumes(
    vector_store,
    directory: str = "resumes",
    client: MCPFilesystemClient | None = None,
) -> int:
    """Index resume files discovered through MCP but absent from Chroma."""
    owns_client = client is None
    client = client or MCPFilesystemClient(root_directory=directory)
    try:
        all_paths = _resume_paths(directory, client)
        known_paths = _indexed_resume_paths(vector_store, directory)
        discovered = client.watch_directory(
            directory,
            known_files=list(known_paths),
        )
        new_paths = {
            str(Path(item["filepath"]).resolve())
            for item in discovered
            if "filepath" in item
        }
        documents = client.batch_process(
            [path for path in all_paths if str(Path(path).resolve()) in new_paths]
        )
        indexed = 0
        for document in documents:
            if document.get("success"):
                create_vector_database_from_documents(document, vector_store)
                indexed += 1
        return indexed
    finally:
        if owns_client:
            client.close()


def read_file_data_and_create_vector_db(
    directory: str = "resumes",
    client: MCPFilesystemClient | None = None,
):
    vector_store = Chroma(
                    collection_name=collection_name,
                    embedding_function=embeddings,
                    persist_directory= persist_directory,
    )
    owns_client = client is None
    client = client or MCPFilesystemClient(root_directory=directory)
    try:
        paths = _resume_paths(directory, client)
        for document in client.batch_process(paths):
            if document.get("success"):
                create_vector_database_from_documents(document, vector_store)
    finally:
        if owns_client:
            client.close()
    return vector_store

def create_vector_database_from_documents(documents, vector_store):

    file_name = documents["filename"]
    print(f"[DEBUG] create_vector_database_from_documents: indexing {file_name}")
    raw_pages = documents.get("content", []) or []
    pages = []
    for page in raw_pages:
        if hasattr(page, "page_content"):
            pages.append(page)
        elif isinstance(page, dict):
            page_content = page.get("page_content") or page.get("text") or ""
            metadata = page.get("metadata") or {}
            pages.append(type("PageLike", (), {"page_content": page_content, "metadata": metadata})())

    if not pages:
        print(f"[DEBUG] create_vector_database_from_documents: no pages for {file_name}")
        return

    resume_text = "\n".join(getattr(page, "page_content", "") for page in pages)
    print(f"[DEBUG] create_vector_database_from_documents: resume_text_length={len(resume_text)}")
    resume_metadata = extract_resume_metadata(resume_text, file_name)
    db_documents = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    for page in pages:
        sections = split_resume_sections(getattr(page, "page_content", ""))

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

    if not db_documents:
        print(f"[DEBUG] create_vector_database_from_documents: no chunks created for {file_name}")
        return

    uuids = [str(uuid4()) for _ in db_documents]
    print(f"[DEBUG] create_vector_database_from_documents: adding {len(db_documents)} chunks")
    vector_store.add_documents(documents=db_documents, ids=uuids)
    # print(f"Added {len(db_documents)} chunks for {file_name}")



def extract_keywords_from_documents(document):
    class DocumentMetadata(BaseModel):
        keywords: list[str] = Field(
            description="5-10 specific keywords, tech terms, or entities found in the text."
        )

    text = getattr(document, "page_content", "") or ""
    try:
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

        extraction = (prompt | structured_llm).invoke({"text": text})
        return extraction.keywords
    except Exception:
        # Fall back to a lightweight regex-based keyword list when the LLM API is
        # unavailable or the OpenRouter account has no remaining credits.
        matches = re.findall(r"[A-Za-z][A-Za-z0-9+/.#-]{2,}", text)
        seen = set()
        keywords = []
        for match in matches:
            candidate = match.strip()
            if len(candidate) < 3 or candidate.lower() in {"the", "and", "with", "that", "this", "from", "have", "been", "into"}:
                continue
            if candidate.lower() not in seen:
                seen.add(candidate.lower())
                keywords.append(candidate)
        return keywords[:10]


def extract_requirements_from_job_description(job_description):
    class DocumentMetadata(BaseModel):
        must_have_skills: list[str] = Field(
            description="List of must have skills, tech terms, or entities found in the text."
        )
        nice_to_have_skills: list[str] = Field(
            description="List of nice to have skills, tech terms, or entities found in the text."
        )

    try:
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
        return {
            "must_have": extraction.must_have_skills,
            "nice_to_have": extraction.nice_to_have_skills,
        }
    except Exception:
        return {"must_have": [], "nice_to_have": []}


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