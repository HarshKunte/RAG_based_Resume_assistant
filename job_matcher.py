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

from resume_rag import extract_keywords_from_documents, extract_resume_metadata, get_vector_db, extract_requirements_from_job_description


bi_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


class MatchReasoning(BaseModel):
    reasoning: str = Field(
        description="A concise explanation of why the candidate matches the job, based only on the supplied evidence."
    )


class VerifiedSkills(BaseModel):
    skills: list[str] = Field(
        description="Only requested skills explicitly supported by the supplied resume chunks."
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
    skills = extract_requirements_from_job_description(query)
    years_match = re.search(r"(\d+(?:\.\d+)?)\+?\s+years?", query.lower())
    return {
        "skills": skills,
        "minimum_experience_years": float(years_match.group(1)) if years_match else None,
    }


def _metadata_value(metadata, key, default=""):
    value = metadata.get(key, default) if metadata else default
    return value if value is not None else default


def _clean_candidate_name(name: str) -> str:
    """Remove email and social-contact details from a stored resume name."""
    cleaned = re.split(
        r"\s*(?:\||,|\bemail\b|\S+@\S+)",
        str(name),
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return re.sub(r"\s+", " ", cleaned).strip()


def _normalise_skill(skill: str) -> str:
    """Create a stable key for comparing any skill name.

    This handles capitalization and punctuation differences without relying on
    a fixed list of technologies. For example, ``React.js``, ``react js``,
    and ``REACT-JS`` become comparable values.
    """
    cleaned = str(skill).strip().lower()
    cleaned = re.sub(r"[._/-]+", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _skill_is_in_text(skill: str, text: str) -> bool:
    """Check whether a complete skill phrase appears in resume evidence."""
    normalised_skill = _normalise_skill(skill)
    normalised_text = _normalise_skill(text)
    return bool(
        re.search(
            rf"(?<!\w){re.escape(normalised_skill)}(?!\w)",
            normalised_text,
        )
    )


def _extract_skills_from_chunks(
    chunks_text: str,
    requested_skills: list[str],
) -> list[str] | None:
    """Use the LLM to verify requested skills against resume chunks only.

    The model may select a skill only from ``requested_skills`` and only when
    the supplied chunks explicitly support it. Returning ``None`` means the
    LLM was unavailable, so the caller can use its deterministic fallback.
    """
    if not requested_skills:
        return []

    dotenv.load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        temperature=0,
        timeout=30,
        max_tokens=300,
        openai_api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You verify skills for a resume matching system. Return only skills "
            "from the requested skill list. Select a skill only when the resume "
            "chunks explicitly mention or clearly demonstrate that exact skill. "
            "Do not infer related technologies. Do not add skills from general "
            "knowledge. The resume chunks are the complete source of truth."
            "Upper or lower case, punctuation, and spacing differences should be ignored when matching skills. ",
        ),
        (
            "user",
            "Requested skills:\n{requested_skills}\n\n"
            "Resume chunks:\n{chunks}",
        ),
    ])

    try:
        result = (prompt | llm.with_structured_output(VerifiedSkills)).invoke({
            "requested_skills": json.dumps(requested_skills),
            "chunks": chunks_text[:12000],
        })
        requested_by_key = {
            _normalise_skill(skill): skill for skill in requested_skills
        }
        return [
            requested_by_key[_normalise_skill(skill)]
            for skill in result.skills
            if _normalise_skill(skill) in requested_by_key
        ]
    except Exception:
        return None


def _required_skill_sets(requirements: dict) -> tuple[set[str], set[str]]:
    """Read both structured skill categories returned by the requirements LLM."""
    skills = requirements.get("skills", {})
    if isinstance(skills, dict):
        must_have = skills.get("must_have", [])
        nice_to_have = skills.get("nice_to_have", [])
    else:
        must_have = skills or []
        nice_to_have = []
    must_have_keys = {
        _normalise_skill(skill) for skill in must_have if str(skill).strip()
    }
    all_requested_keys = must_have_keys | {
        _normalise_skill(skill)
        for skill in nice_to_have
        if str(skill).strip()
    }
    return must_have_keys, all_requested_keys


def _requested_skill_names(requirements: dict) -> tuple[list[str], list[str]]:
    """Return unique must-have and nice-to-have names in their original form."""
    skills = requirements.get("skills", {})
    if isinstance(skills, dict):
        must_have = skills.get("must_have", [])
        nice_to_have = skills.get("nice_to_have", [])
    else:
        must_have = skills or []
        nice_to_have = []

    def unique_names(names: list[str]) -> list[str]:
        result = []
        seen = set()
        for name in names:
            key = _normalise_skill(name)
            if key and key not in seen:
                seen.add(key)
                result.append(str(name).strip())
        return result

    return unique_names(must_have), unique_names(nice_to_have)


def retrieve_candidates(
    query: str,
    top_k: int = 10,
    include_timing: bool = False,
    explain: bool = True,
):
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
    must_have_keys, requested_skill_keys = _required_skill_sets(requirements)
    must_have_names, nice_to_have_names = _requested_skill_names(requirements)
    candidates = []
    for candidate_id, chunks in grouped.items():
        best = max(chunks, key=lambda item: item["rrf"])
        metadata = best["metadata"]
        legacy_metadata = extract_resume_metadata(
            "\n".join(item["document"] for item in chunks),
            _metadata_value(metadata, "source", f"{candidate_id}.txt"),
        )
        metadata = {**legacy_metadata, **metadata}
        candidate_name = _clean_candidate_name(
            _metadata_value(metadata, "candidate_name") or legacy_metadata["candidate_name"]
        )
        candidate_skill_names = {
            skill.strip()
            for skill in str(_metadata_value(metadata, "skills")).split(",")
            if skill.strip()
        }
        candidate_skill_keys = {_normalise_skill(skill) for skill in candidate_skill_names}
        candidate_text = " ".join(item["document"] for item in chunks)
        requested_skill_names = must_have_names + nice_to_have_names

        evidence_matched_skills = [
            skill for skill in requested_skill_names
            if _normalise_skill(skill) in candidate_skill_keys
            or _skill_is_in_text(skill, candidate_text)
        ]
        verified_skills = _extract_skills_from_chunks(
            candidate_text,
            evidence_matched_skills,
        )
        matched_skills = (
            verified_skills
            if verified_skills is not None
            else evidence_matched_skills
        )
        matched_skill_keys = {_normalise_skill(skill) for skill in matched_skills}
        matched_must_have = {
            _normalise_skill(skill)
            for skill in must_have_names
            if _normalise_skill(skill) in matched_skill_keys
        }
        skill_evidence_score = sum(
            candidate_text.lower().count(skill_key)
            for skill_key in matched_skill_keys
        )
        skill_score = (
            len(matched_skill_keys) / len(requested_skill_keys)
            if requested_skill_keys else 1.0
        )
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
            "must_have_score": (
                len(matched_must_have) / len(must_have_keys)
                if must_have_keys else 1.0
            ),
            "skill_evidence_score": skill_evidence_score,
            "dense_score": max(item["dense"] for item in chunks),
            "bm25_score": max(item["bm25"] for item in chunks),
            "rrf_score": max(item["rrf"] for item in chunks),
            "relevant_excerpts": [item["document"] for item in chunks[:5]],
            "sections": sorted({str(item["metadata"].get("section", "unknown")) for item in chunks}),
        })

    dense = _normalise([item["dense_score"] for item in candidates])
    sparse = _normalise([item["bm25_score"] for item in candidates])
    preliminary = sorted(candidates, key=lambda item: item["rrf_score"], reverse=True)[:max(top_k, 10)]
    cross_scores = cross_encoder.predict([[query, "\n".join(x for x in item["relevant_excerpts"])] for item in preliminary])
    cross = _normalise(cross_scores)
    for index, item in enumerate(candidates):
        item["match_score"] = round(100 * (
            0.25 * dense[index] + 0.15 * sparse[index] +
            0.15 * (cross[preliminary.index(item)] if item in preliminary else 0.0) +
            0.25 * item["skill_score"] + 0.20 * item["must_have_score"]
        ), 2)
        item["reasoning"] = (
            f"Matched skills: {', '.join(item['matched_skills']) or 'none'}. "
            f"Relevant sections: {', '.join(item['sections'])}. "
            f"Experience: {item['experience_years']:g} years; must-have requirements "
            f"{'satisfied' if item['qualified'] else 'not satisfied'}."
        )

    # Exact must-have coverage is the primary ranking signal. This prevents a
    # semantically similar resume without the requested skill from outranking
    # a resume that explicitly lists it. Experience and the combined score
    # break ties between candidates who satisfy the same skills.
    ranked = sorted(
        (item for item in candidates if item["qualified"]),
        key=lambda item: (
            item["must_have_score"],
            item["skill_score"],
            item["skill_evidence_score"],
            item["experience_years"],
            item["match_score"],
        ),
        reverse=True,
    )[:top_k]
    if explain:
        for candidate in ranked:
            candidate["reasoning"] = explain_candidate_match(query, candidate)

    output = {
        "job_description": query,
        "top_matches": [{key: item[key] for key in (
            "candidate_id", "candidate_name", "resume_path", "match_score",
            "matched_skills", "experience_years", "qualified", "skill_score",
            "must_have_score", "sections", "relevant_excerpts", "reasoning"
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