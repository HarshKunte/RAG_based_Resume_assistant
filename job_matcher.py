import json
import os
import re
import time
from collections import defaultdict

import dotenv
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer, util
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from resume_rag import extract_keywords_from_documents, extract_resume_metadata, get_vector_db


bi_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


class MatchReasoning(BaseModel):
    reasoning: str = Field(
        description="A concise explanation of why the candidate matches the job, based only on the supplied evidence."
    )


def explain_candidate_match(query: str, candidate: dict) -> str:
    """Use an LLM to explain a score that was calculated by the retrieval pipeline."""
    dotenv.load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return candidate["reasoning"]

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a resume matching assistant. Explain the candidate's fit using only the evidence provided. "
            "Do not change the numeric score, invent experience, or claim a skill that is not listed. "
            "Mention matched skills, relevant resume sections, experience, and any unmet must-have requirement. "
            "Keep the explanation concise and professional.",
        ),
        (
            "user",
            "Job description:\n{query}\n\nCandidate evidence:\n{evidence}",
        ),
    ])
    evidence = json.dumps({
        "candidate_name": candidate["candidate_name"],
        "match_score": candidate["match_score"],
        "matched_skills": candidate["matched_skills"],
        "experience_years": candidate["experience_years"],
        "sections": candidate["sections"],
        "relevant_excerpts": candidate["relevant_excerpts"],
        "qualified": candidate["qualified"],
    })
    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        temperature=0,
        openai_api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )
    try:
        result = (prompt | llm.with_structured_output(MatchReasoning)).invoke({
            "query": query,
            "evidence": evidence,
        })
        return result.reasoning
    except Exception:
        return candidate["reasoning"]


def ranking(scores):
    return list(np.argsort(-scores))


def reciprocal_rank_fusion(rankings, k=60):
    fused_scores = defaultdict(float)
    for result_ranking in rankings:
        for rank, document_index in enumerate(result_ranking):
            fused_scores[document_index] += 1.0 / (k + rank + 1)
    return dict(fused_scores)


def _normalise(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    if np.max(values) == np.min(values):
        return np.ones(len(values))
    return (values - np.min(values)) / (np.max(values) - np.min(values))


def extract_job_requirements(query: str) -> dict:
    query_lower = query.lower()
    query_document = type("TextDocument", (), {"page_content": query})()
    skills = extract_keywords_from_documents(query_document)
    years_match = re.search(r"(\d+(?:\.\d+)?)\+?\s+years?", query_lower)
    return {
        "skills": skills,
        "minimum_experience_years": float(years_match.group(1)) if years_match else None,
    }


def _metadata_value(metadata, key, default=""):
    value = metadata.get(key, default) if metadata else default
    return value if value is not None else default


def retrieve_candidates(query: str, top_k: int = 10, include_timing: bool = False):
    started = time.perf_counter()
    stored = get_vector_db().get(include=["documents", "metadatas"])
    documents = stored.get("documents", [])
    metadatas = stored.get("metadatas", [])
    if not documents:
        return {"job_description": query, "top_matches": [], "timing_ms": {"total": 0.0}}

    query_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    document_embeddings = bi_encoder.encode(documents, convert_to_tensor=True)
    dense_scores = util.cos_sim(query_embedding, document_embeddings)[0].cpu().numpy()
    bm25 = BM25Okapi([doc.lower().split() for doc in documents])
    bm25_scores = np.asarray(bm25.get_scores(query.lower().split()))
    fused = reciprocal_rank_fusion([ranking(bm25_scores), ranking(dense_scores)])

    grouped = defaultdict(list)
    for index, rrf_score in fused.items():
        metadata = metadatas[index] or {}
        source = _metadata_value(metadata, "source", f"candidate_{index}")
        candidate_id = _metadata_value(metadata, "candidate_id") or re.sub(
            r"[^a-z0-9]+", "_", source.lower()
        ).strip("_")
        grouped[candidate_id].append({
            "document": documents[index], "metadata": metadata,
            "dense": float(dense_scores[index]), "bm25": float(bm25_scores[index]),
            "rrf": float(rrf_score),
        })

    requirements = extract_job_requirements(query)
    candidates = []
    for candidate_id, chunks in grouped.items():
        best = max(chunks, key=lambda item: item["rrf"])
        metadata = best["metadata"]
        legacy_metadata = extract_resume_metadata(
            "\n".join(item["document"] for item in chunks),
            _metadata_value(metadata, "source", f"{candidate_id}.txt"),
        )
        metadata = {**legacy_metadata, **metadata}
        candidate_name = _metadata_value(metadata, "candidate_name") or legacy_metadata["candidate_name"]
        candidate_skills = {skill.strip() for skill in str(_metadata_value(metadata, "skills")).split(",") if skill.strip()}
        candidate_text = " ".join(item["document"] for item in chunks)
        candidate_skills.update(extract_keywords_from_documents(
            type("TextDocument", (), {"page_content": candidate_text})()
        ))
        matched_skills = sorted(candidate_skills.intersection(requirements["skills"]))
        skill_score = len(matched_skills) / len(requirements["skills"]) if requirements["skills"] else 1.0
        experience = float(_metadata_value(metadata, "experience_years", 0) or 0)
        minimum_years = requirements["minimum_experience_years"]
        qualified = minimum_years is None or experience >= minimum_years
        chunks = sorted(chunks, key=lambda item: item["rrf"], reverse=True)
        candidates.append({
            "candidate_id": candidate_id,
            "candidate_name": candidate_name or candidate_id,
            "resume_path": _metadata_value(metadata, "source"),
            "experience_years": experience,
            "matched_skills": matched_skills,
            "qualified": qualified,
            "skill_score": skill_score,
            "must_have_score": 1.0 if qualified else 0.0,
            "dense_score": max(item["dense"] for item in chunks),
            "bm25_score": max(item["bm25"] for item in chunks),
            "rrf_score": max(item["rrf"] for item in chunks),
            "relevant_excerpts": [item["document"][:500] for item in chunks[:3]],
            "sections": sorted({str(item["metadata"].get("section", "unknown")) for item in chunks}),
        })

    dense = _normalise([item["dense_score"] for item in candidates])
    sparse = _normalise([item["bm25_score"] for item in candidates])
    preliminary = sorted(candidates, key=lambda item: item["rrf_score"], reverse=True)[:max(top_k, 10)]
    cross_scores = cross_encoder.predict([[query, item["relevant_excerpts"][0]] for item in preliminary])
    cross = _normalise(cross_scores)
    for index, item in enumerate(candidates):
        item["match_score"] = round(100 * (
            0.35 * dense[index] + 0.25 * sparse[index] +
            0.20 * (cross[preliminary.index(item)] if item in preliminary else 0.0) +
            0.10 * item["skill_score"] + 0.10 * item["must_have_score"]
        ), 2)
        item["reasoning"] = (
            f"Matched skills: {', '.join(item['matched_skills']) or 'none'}. "
            f"Relevant sections: {', '.join(item['sections'])}. "
            f"Experience: {item['experience_years']:g} years; must-have requirements "
            f"{'satisfied' if item['qualified'] else 'not satisfied'}."
        )

    ranked = sorted((item for item in candidates if item["qualified"]), key=lambda item: item["match_score"], reverse=True)[:top_k]
    for candidate in ranked:
        candidate["reasoning"] = explain_candidate_match(query, candidate)

    output = {
        "job_description": query,
        "top_matches": [{key: item[key] for key in (
            "candidate_name", "resume_path", "match_score", "matched_skills",
            "relevant_excerpts", "reasoning"
        )} for item in ranked],
    }
    if include_timing:
        output["timing_ms"] = {"total": round((time.perf_counter() - started) * 1000, 2)}
    return output


def get_top_matches(query: str, top_k: int = 10):
    return retrieve_candidates(query, top_k=top_k, include_timing=True)


if __name__ == "__main__":
    print(json.dumps(get_top_matches(
        "Required candidate should have 4+ years of professional experience in React"
    ), indent=2))