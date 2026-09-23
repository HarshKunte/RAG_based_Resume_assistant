"""LangGraph orchestration for the RAG resume matching application.

The matching and RAG implementations remain in ``job_matcher`` and
``resume_rag``. This module owns state, conversation intent, and the graph
that connects those capabilities.
"""

import os
import re
from typing import Any, TypedDict

import dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from mcp_client import MCPFilesystemClient


class State(TypedDict, total=False):
    query: str
    conversation_history: list[dict[str, str]]
    job_requirement_understanding: dict[str, Any]
    shortlisted_candidates: list[dict[str, Any]]
    previous_shortlist: list[dict[str, Any]]
    screening_rounds: dict[str, Any]
    report: str
    feedback: str
    action: str
    next_step: str


def _matcher():
    """Import the heavy retrieval stack only when a graph run needs it."""
    from job_matcher import extract_job_requirements, retrieve_candidates

    return extract_job_requirements, retrieve_candidates


def _query_text(query: str) -> str:
    """Remove conversational wrappers while preserving the JD content."""
    cleaned = re.sub(r"(?is)^.*?job description\s*[-:]", "", query).strip()
    return cleaned or query.strip()


def _extract_job_description_with_llm(query: str) -> str:
    """Use an LLM to separate the job description from the user's request.

    A local extraction fallback is kept because the agent should still be
    usable when the API key is missing or the LLM request fails.
    """
    dotenv.load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    fallback = _query_text(query)

    if not api_key:
        return fallback

    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        temperature=0,
        max_tokens=500,
        timeout=30,
        openai_api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )
    messages = [
        SystemMessage(
            content=(
                "You extract job descriptions for a resume matching system. "
                "Return only the job description text. Remove conversational "
                "phrases such as 'find candidates' or 'suggest candidates'. "
                "Do not invent or rewrite requirements. If the message itself "
                "contains the requirements, preserve them exactly."
            )
        ),
        HumanMessage(content=query),
    ]

    try:
        response = llm.invoke(messages)
        extracted_text = response.content.strip()
        return extracted_text or fallback
    except Exception:
        return fallback


def _detect_action(query: str) -> str:
    lowered = query.lower()
    if "interview" in lowered or "screening question" in lowered:
        return "interview_questions"
    if "compare" in lowered or "side by side" in lowered:
        return "compare"
    if "why did" in lowered or "why is" in lowered or "why has" in lowered and "rank" in lowered:
        return "explain_ranking"
    if "generate report" in lowered or "detailed report" in lowered:
        return "report"
    if "give" in lowered or "find" in lowered or "show" in lowered or "search" in lowered or "look for" in lowered or "find candidates" in lowered or "suggest candidates" in lowered:
        return "search"
    return "ask_llm"


def decide_action(state: State) -> dict[str, Any]:
    """Choose the next workflow path from the user's current query.

    Comparison, interview, and ranking questions can use the shortlist already
    stored in state. A first-time request for one of those actions still goes
    through retrieval once because there are no candidates to inspect yet.
    """
    query = state.get("query", "").strip()
    history = list(state.get("conversation_history", []))
    history.append({"role": "user", "content": query})
    action = _detect_action(query)
    # print(f"Decided action: {action}")
    has_shortlist = bool(state.get("shortlisted_candidates"))
    needs_retrieval = action == "search" or not has_shortlist
    return {
        "action": action,
        "next_step": "parse_jd" if needs_retrieval else "generate_report",
        "conversation_history": history,
    }


def route_after_decision(state: State) -> str:
    """Return the graph node selected by ``decide_action``."""
    return state.get("next_step", "parse_jd")


def parse_jd(state: State) -> dict[str, Any]:
    query = state.get("query", "").strip()
    print(f"[DEBUG] parse_jd: starting for query='{query}'")
    job_description = _extract_job_description_with_llm(query)
    print(f"[DEBUG] parse_jd: extracted job_description='{job_description[:120]}...'")
    return {
        "job_requirement_understanding": {
            **state.get("job_requirement_understanding", {}),
            "job_description": job_description,
        },
        "action": _detect_action(query),
    }


def extract_requirements(jd: str) -> dict[str, Any]:
    """Return must-have, nice-to-have, and experience requirements."""
    extract_job_requirements, _ = _matcher()
    extracted = extract_job_requirements(jd)
    skills = extracted.get("skills", {})
    if isinstance(skills, dict):
        must_have = skills.get("must_have", [])
        nice_to_have = skills.get("nice_to_have", [])
    else:
        must_have = skills or []
        nice_to_have = []
    return {
        "must_have": must_have,
        "nice_to_have": nice_to_have,
        "minimum_experience_years": extracted.get("minimum_experience_years"),
    }


def extract_requirements_node(state: State) -> dict[str, Any]:
    print("[DEBUG] extract_requirements_node: beginning requirement extraction")
    jd = state["job_requirement_understanding"].get("job_description", "")
    print(f"[DEBUG] extract_requirements_node: jd_length={len(jd)}")
    requirements = extract_requirements(jd)
    print(f"[DEBUG] extract_requirements_node: requirements={requirements}")
    return {"job_requirement_understanding": {
        **state.get("job_requirement_understanding", {}),
        **requirements,
    }}


def search_resumes(state: State) -> dict[str, Any]:
    _, retrieve_candidates = _matcher()
    requirements = state["job_requirement_understanding"]
    query = requirements["job_description"]
    print(f"[DEBUG] search_resumes: searching for query='{query}'")
    try:
        search_result = retrieve_candidates(query, top_k=10, explain=False)
    except TypeError:
        print("[DEBUG] search_resumes: falling back to legacy retrieve_candidates signature")
        search_result = retrieve_candidates(query, top_k=10)
    matches = search_result.get("top_matches", [])
    print(f"[DEBUG] search_resumes: matched_count={len(matches)}")
    return {
        "previous_shortlist": state.get("shortlisted_candidates", []),
        "shortlisted_candidates": matches,
    }


def rag_search(query: str, top_k: int = 10) -> list[dict[str, Any]]:
    """Expose the existing hybrid RAG matcher as an agent-callable tool."""
    _, retrieve_candidates = _matcher()
    return retrieve_candidates(query, top_k=top_k).get("top_matches", [])


def get_agent_tools() -> dict[str, Any]:
    """Return MCP filesystem, RAG, and specialist tools available to the agent."""
    filesystem = MCPFilesystemClient()

    return {
        "list_files": filesystem.list_files,
        "read_file": filesystem.read_file,
        "watch_directory": filesystem.watch_directory,
        "batch_process": filesystem.batch_process,
        "rag_search": rag_search,
        "extract_requirements": extract_requirements,
        "compare_candidates": compare_candidates,
        "generate_interview_questions": generate_interview_questions,
    }


def rank_candidates(state: State) -> dict[str, Any]:
    ranked = sorted(
        state.get("shortlisted_candidates", []),
        key=lambda candidate: candidate.get("match_score", 0),
        reverse=True,
    )
    deep_analysis = [{
        "candidate_id": candidate.get("candidate_id", candidate.get("candidate_name", "")),
        "strengths": candidate.get("matched_skills", []),
        "gaps": [] if candidate.get("qualified", True) else ["Minimum experience requirement"],
        "evidence": candidate.get("relevant_excerpts", [])[:3],
    } for candidate in ranked[:3]]
    final_recommendations = [{
        "candidate_id": candidate.get("candidate_id", candidate.get("candidate_name", "")),
        "recommendation": "hire" if candidate.get("qualified", True) and candidate.get("match_score", 0) >= 60 else "no-hire",
    } for candidate in ranked[:3]]
    return {
        "shortlisted_candidates": ranked,
        "screening_rounds": {
            "initial": ranked[:10],
            "deep_analysis": deep_analysis,
            "final_recommendations": final_recommendations,
        },
    }


def compare_candidates(
    candidate_ids: list[str],
    candidates: list[dict[str, Any]] | None = None,
) -> str:
    # print(f"Line 229: candidate_ids = {candidate_ids} candidates = {candidates}")
    """Create a deterministic side-by-side comparison from retrieved evidence."""
    candidates = candidates or []
    wanted = {candidate_id.lower() for candidate_id in candidate_ids}
    selected = [candidate for candidate in candidates if any(
        value.lower() in wanted
        for value in (candidate.get("candidate_id", ""), candidate.get("candidate_name", ""))
    )]
    if not selected:
        return "No requested candidates were found in the current shortlist."
    rows = ["Candidate | Score | Skills | Evidence"]
    rows.append("--- | ---: | --- | ---")
    for candidate in selected:
        rows.append(
            f"{candidate.get('candidate_name', 'Unknown')} | "
            f"{candidate.get('match_score', 0):g} | "
            f"{', '.join(candidate.get('matched_skills', [])) or 'None'} | "
            f"{candidate.get('reasoning', 'No reasoning available')}"
        )
    return "\n".join(rows)


def generate_interview_questions(
    candidate_id: str,
    candidates: list[dict[str, Any]] | None = None,
    required_skills: list[str] | None = None,
) -> list[str]:
    # print(f"Name or ID: {candidate_id}, Required skills: {required_skills}")
    """Generate LLM interview questions when the candidate has all required skills."""
    candidate = next((item for item in candidates or [] if re.search(rf"\b{candidate_id.lower()}\b", item.get("candidate_id", "").lower(), re.IGNORECASE) or re.search(rf"\b{candidate_id.lower()}\b", item.get("candidate_name", "").lower(), re.IGNORECASE)), None)
    if not candidate:
        return [f"No candidate named '{candidate_id}' is in the current shortlist."]
    skills = candidate.get("matched_skills", [])
    required_skills = required_skills or []
    matched_skill_names = {skill.lower() for skill in skills}
    has_required_skills = all(
        skill.lower() in matched_skill_names for skill in required_skills
    )

    if has_required_skills and required_skills:
        dotenv.load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            llm = ChatOpenAI(
                model="openai/gpt-4o-mini",
                temperature=0.3,
                max_tokens=400,
                timeout=30,
                openai_api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
            )
            prompt = (
                "Create 5 practical technical screening questions for this candidate. "
                "Questions must test real experience, decision-making, and problem solving. "
                "Use only the listed skills and resume evidence. Return one question per line.\n\n"
                f"Candidate: {candidate.get('candidate_name', candidate_id)}\n"
                f"Required skills: {', '.join(required_skills)}\n"
                f"Matched skills: {', '.join(skills)}\n"
                f"Resume evidence: {' '.join(candidate.get('relevant_excerpts', []))}"
            )
            try:
                response = llm.invoke([HumanMessage(content=prompt)])
                questions = [
                    re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
                    for line in response.content.splitlines()
                    if line.strip()
                ]
                if questions:
                    evidence_question = (
                        f"Describe your most recent production work using "
                        f"{skills[0]}."
                    )
                    if not any("production work" in question.lower() for question in questions):
                        if len(questions) >= 5:
                            questions[-1] = evidence_question
                        else:
                            questions.append(evidence_question)
                    return questions[:5]
            except Exception:
                pass

    questions = [
        f"Describe your most recent production work using {skill}."
        for skill in skills[:3]
    ]
    questions.append(
        "Walk through a project that demonstrates the experience claimed on your resume."
    )
    if not candidate.get("qualified", True):
        questions.append(
            "How would you close the gap against the role's minimum requirements?"
        )
    return questions


def generate_report(state: State) -> dict[str, Any]:
    candidates = state.get("shortlisted_candidates", [])
    action = state.get("action", "search")
    query = state.get("query", "")
    print(f"[DEBUG] generate_report: action={action}, candidate_count={len(candidates)}, query='{query}'")
    candidate_names = [candidate.get("candidate_name", candidate.get("candidate_id", "")) for candidate in candidates]
    split_candidate_names = [part for name in candidate_names for part in name.split()]
    if action == "compare":
        # print(f"Line 330: candidate_names = {candidate_names}")
        top_count = re.search(r"top\s+(\d+)", query.lower())
        if top_count:
            selected = candidates[:int(top_count.group(1))]
            report = compare_candidates(
                [candidate.get("candidate_id", candidate.get("candidate_name", "")) for candidate in selected],
                candidates,
            )
        elif any(sub.lower() in query.lower() for sub in split_candidate_names):
            query_lower = query.lower()
            ids = []
            for candidate in candidates:
                name = str(candidate.get("candidate_name", "")).strip()
                parts = name.split()
                if not parts:
                    continue
                if any(part.lower() in query_lower for part in parts):
                    ids.append(candidate.get("candidate_id", name))
            if not ids:
                ids = re.findall(r"[A-Za-z][A-Za-z -]+", query)
            report = compare_candidates(ids[-3:] if ids else [], candidates)
        else:
            # print(f"Line 332: No matching candidate names found in query, extracting IDs")
            ids = re.findall(r"[A-Za-z][A-Za-z -]+", query)
            report = compare_candidates(ids[-3:], candidates)
    elif action == "interview_questions":
        name = query.split("for", 1)[-1].strip(" :?.")
        required_skills = state.get("job_requirement_understanding", {}).get(
            "must_have", []
        )
        report = "\n".join(
            f"- {question}"
            for question in generate_interview_questions(
                name,
                candidates,
                required_skills,
            )
        )
    elif action == "explain_ranking" and len(candidates) >= 2:
        first, second = candidates[:2]
        report = (
            f"{first.get('candidate_name')} ranked higher with a score of {first.get('match_score', 0):g} "
            f"versus {second.get('match_score', 0):g}. Their matched skills are "
            f"{', '.join(first.get('matched_skills', [])) or 'none'}, compared with "
            f"{', '.join(second.get('matched_skills', [])) or 'none'} for {second.get('candidate_name')}."
        )
    elif action == "ask_llm":
        dotenv.load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        fallback = "No API Key found. Please set OPENAI_API_KEY in your environment to enable LLM responses."
        
        if not api_key:
            return fallback
        history = list(state.get("conversation_history", []))
        llm = ChatOpenAI(
            model="openai/gpt-4o-mini",
            temperature=0.3,
            openai_api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            max_tokens=400,
        )

        llm_prompt = (
            "You are an expert at analyzing job descriptions and resumes."

            f"Here is the previous conversation history:\n {history}\n\n"
            f"Here are the top candidates retrieved:\n {candidates}\n\n"

            f"On above data answer this query:\n {query}\n\n"
        )
        llm_response = llm.invoke([SystemMessage(content=llm_prompt)])
        report = llm_response.content
    else:
        lines = ["Top matches:"]
        for index, candidate in enumerate(candidates, start=1):
            lines.append(
                f"{index}. {candidate.get('candidate_name')} - {candidate.get('match_score', 0):g}/100. "
                f"{candidate.get('reasoning', '')}"
            )
        report = "\n".join(lines) if candidates else "No matching candidates were found."
    previous = {
        candidate.get("candidate_id", candidate.get("candidate_name", "")): index
        for index, candidate in enumerate(state.get("previous_shortlist", []), start=1)
    }
    if previous and candidates:
        changes = []
        for index, candidate in enumerate(candidates[:3], start=1):
            candidate_id = candidate.get("candidate_id", candidate.get("candidate_name", ""))
            if candidate_id in previous and previous[candidate_id] != index:
                changes.append(f"{candidate.get('candidate_name')} moved from #{previous[candidate_id]} to #{index}")
        if changes:
            report += "\n\nRanking changes after refinement:\n" + "\n".join(f"- {change}" for change in changes)
    conversation_history = list(state.get("conversation_history", []))
    conversation_history.append({"role": "assistant", "content": report})
    return {"report": report, "conversation_history": conversation_history}


def human_feedback_loop(state: State) -> dict[str, Any]:
    feedback = state.get("feedback", "").strip()
    history = [
        *state.get("conversation_history", []),
        {"role": "assistant", "content": state.get("report", "")},
    ]
    if feedback:
        history.append({"role": "user", "content": feedback})
    return {"conversation_history": history}


def build_workflow():
    workflow = StateGraph(State)
    workflow.add_node("decide_action", decide_action)
    workflow.add_node("parse_jd", parse_jd)
    workflow.add_node("extract_requirements", extract_requirements_node)
    workflow.add_node("search_resumes", search_resumes)
    workflow.add_node("rank_candidates", rank_candidates)
    workflow.add_node("generate_report", generate_report)
    workflow.add_node("human_feedback_loop", human_feedback_loop)
    workflow.add_edge(START, "decide_action")
    workflow.add_conditional_edges(
        "decide_action",
        route_after_decision,
        {
            "parse_jd": "parse_jd",
            "generate_report": "generate_report",
        },
    )
    workflow.add_edge("parse_jd", "extract_requirements")
    workflow.add_edge("extract_requirements", "search_resumes")
    workflow.add_edge("search_resumes", "rank_candidates")
    workflow.add_edge("rank_candidates", "generate_report")
    workflow.add_edge("generate_report", END)
    return workflow.compile()


def run_query(
    app,
    query: str,
    previous: State | None = None,
    progress_callback=None,
) -> State:
    """Run one query, optionally reporting each completed graph node."""
    previous = previous or {}
    input_state = {
        **previous,
        "query": query,
        "feedback": "",
        "conversation_history": previous.get("conversation_history", []),
    }

    if progress_callback is None:
        return app.invoke(input_state)

    final_state = dict(input_state)
    for update in app.stream(input_state, stream_mode="updates"):
        for node_name, node_update in update.items():
            final_state.update(node_update)
            progress_callback(node_name)

    return final_state


def cli() -> None:
    app = build_workflow()
    state: State = {"conversation_history": [], "job_requirement_understanding": {}}
    # print("Resume matching assistant. Type 'exit' to quit.")
    while True:
        query = input("\nYou: ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        state = run_query(app, query, state)
        # print(f"\nAssistant:\n{state.get('report', '')}")


if __name__ == "__main__":
    cli()
