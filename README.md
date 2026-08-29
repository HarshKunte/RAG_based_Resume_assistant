# RAG-based Resume Assistant

This project builds a resume matching pipeline powered by Retrieval-Augmented Generation (RAG). It reads resume files from the `resumes/` folder, chunks and embeds them with sentence-transformers, stores them in Chroma, and then ranks candidates against a job description using both semantic search and BM25 retrieval.

The system is designed to help match job descriptions to relevant resumes and explain why a candidate is a good fit.

## What this project does

- Loads resumes from `resumes/`
- Splits resume text into sections and chunks
- Creates a vector database with Chroma
- Uses Hugging Face embeddings for semantic retrieval
- Ranks candidate resumes against a job description
- Provides reasoning/explanations for top matches
- Evaluates retrieval quality using standard ranking metrics

## Project structure

- `resume_rag.py` – creates and loads the vector database
- `job_matcher.py` – retrieves and ranks candidate matches for a job description
- `evaluation.py` – evaluates retrieval performance on sample jobs
- `evaluation_data/` – evaluation inputs and relevance labels
- `resumes/` – candidate resume files
- `chroma_langchain_db/` – persisted Chroma database

## Requirements

- Python 3.10+
- pip
- An OpenAI-compatible API key for LLM-based keyword extraction and explanation (stored in `.env`)

## Installation

1. Open a terminal in the project root.
2. Create a virtual environment:

```bash
python -m venv .venv
```

3. Activate the virtual environment:

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Create a `.env` file in the root folder and add your API key:

```env
OPENAI_API_KEY=your_api_key_here
```

> This project uses `langchain_openai` with an OpenRouter-compatible endpoint, so your key should be valid for that setup.

## Running the app

### Build or refresh the resume database

```bash
python resume_rag.py
```

This script loads resumes and creates or refreshes the Chroma database under `chroma_langchain_db/`.

### Run the job matching pipeline

```bash
python job_matcher.py
```

This script executes a sample job match query and prints top candidates and scores.

### Evaluate the system

```bash
python evaluation.py
```

This runs the evaluation workflow and prints metrics like recall, precision, MRR, and NDCG.

## Example usage

You can edit the query inside `job_matcher.py` or call `get_top_matches()` from another Python script:

```python
from job_matcher import get_top_matches

result = get_top_matches("Required candidate should have 4+ years of experience in React and Python", top_k=5)
print(result)
```

## Notes

- The first run may take some time because embeddings and model initialization can be heavy.
- The project depends on downloaded models from Hugging Face and may require internet access.
- If the Chroma database already exists, the script will reuse it instead of recreating it.

## Troubleshooting

- If you get a missing API key error, make sure `.env` exists and `OPENAI_API_KEY` is set correctly.
- If a model download fails, ensure your internet connection is working and the Python environment is activated.
- If Chroma errors appear, remove the `chroma_langchain_db/` folder and rerun `python resume_rag.py`.
