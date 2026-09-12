"""Streamlit user interface for the RAG resume matching assistant."""

import streamlit as st

from matching_agent import (
    build_workflow,
    compare_candidates,
    generate_interview_questions,
    run_query,
)


st.set_page_config(
    page_title="Talent Atlas | Resume Intelligence",
    page_icon="TA",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #17231f;
        --muted: #68736f;
        --line: #dce4df;
        --paper: #f7faf7;
        --mint: #dcefe6;
        --green: #1b6b52;
        --coral: #ed725b;
        --gold: #e6b94f;
    }

    html, body, [class*="css"] {
        font-family: 'Manrope', sans-serif;
        color: var(--ink);
    }

    .stApp {
        background:
            radial-gradient(circle at 92% 0%, rgba(230,185,79,.16), transparent 23rem),
            linear-gradient(135deg, #f7faf7 0%, #f1f6f1 52%, #fbf8f2 100%);
    }

    [data-testid="stSidebar"] {
        background: #173d32;
        border-right: 0;
    }

    [data-testid="stSidebar"] * { color: #edf8f1; }
    [data-testid="stSidebar"] .stCaption { color: #b4ccc0; }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.16); }
    [data-testid="stSidebar"] .stButton > button {
        color: #173d32;
        background: #f4c85d;
        border: 0;
    }

    .block-container { max-width: 1380px; padding-top: 2.2rem; }
    [data-testid="stAppViewContainer"] h1,
    [data-testid="stAppViewContainer"] h2,
    [data-testid="stAppViewContainer"] h3,
    [data-testid="stAppViewContainer"] h4,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] li,
    [data-testid="stAppViewContainer"] label {
        color: var(--ink) !important;
    }
    [data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] p {
        color: var(--muted) !important;
    }
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] {
        color: var(--ink) !important;
    }
    [data-testid="stAppViewContainer"] table,
    [data-testid="stAppViewContainer"] th,
    [data-testid="stAppViewContainer"] td {
        color: var(--ink) !important;
        border-color: var(--line) !important;
    }
    [data-testid="stAppViewContainer"] [data-testid="stStatusWidget"],
    [data-testid="stAppViewContainer"] [data-testid="stStatusWidget"] * {
        color: var(--ink) !important;
    }
    [data-testid="stChatInput"], [data-testid="stChatInputContainer"] {
        background: rgba(255,255,255,.96);
        border: 1px solid #cbd8d0;
        border-radius: 8px;
    }
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInputContainer"] textarea,
    [data-testid="stAppViewContainer"] textarea,
    [data-baseweb="textarea"] textarea,
    textarea[placeholder] {
        color: var(--ink) !important;
        -webkit-text-fill-color: var(--ink) !important;
        background: transparent !important;
    }
    [data-testid="stChatInput"] textarea::placeholder,
    [data-testid="stChatInputContainer"] textarea::placeholder {
        color: #697771 !important;
        opacity: 1;
    }
    [data-baseweb="textarea"] textarea::placeholder,
    textarea[placeholder]::placeholder {
        color: #697771 !important;
        -webkit-text-fill-color: #697771 !important;
        opacity: 1 !important;
    }
    [data-testid="stChatInput"] button { color: var(--green) !important; }
    .brand-mark {
        display: flex; align-items: center; gap: 12px; margin-bottom: 2rem;
        font-weight: 800; letter-spacing: -.04em; font-size: 1.15rem;
    }
    .brand-dot {
        width: 32px; height: 32px; display: grid; place-items: center;
        background: #f4c85d; color: #173d32; border-radius: 9px;
        font-family: 'DM Mono', monospace; font-size: .72rem; font-weight: 500;
    }
    .eyebrow {
        color: var(--coral); text-transform: uppercase; letter-spacing: .15em;
        font: 500 .72rem 'DM Mono', monospace; margin-bottom: .55rem;
    }
    h1 { font-size: clamp(2rem, 4vw, 4.15rem); line-height: .98; letter-spacing: -.075em; margin: 0; }
    h2, h3 { letter-spacing: -.045em; }
    .hero-copy { color: var(--muted); max-width: 650px; margin: 1rem 0 1.8rem; font-size: 1.04rem; line-height: 1.65; }
    .hero-rule { height: 1px; background: var(--line); margin: 1.3rem 0 1.8rem; }
    .section-label {
        color: var(--muted); text-transform: uppercase; letter-spacing: .13em;
        font: 500 .68rem 'DM Mono', monospace; margin: 1.2rem 0 .7rem;
    }
    .candidate-card {
        background: rgba(255,255,255,.8); border: 1px solid var(--line);
        border-radius: 8px; padding: 1.15rem 1.25rem; margin-bottom: .75rem;
        box-shadow: 0 8px 28px rgba(23,61,50,.045); color: var(--ink) !important;
    }
    .candidate-top { display: flex; justify-content: space-between; gap: 1rem; align-items: start; }
    .candidate-name { font-size: 1.12rem; font-weight: 800; letter-spacing: -.04em; }
    .candidate-meta { color: #52615a !important; font-size: .82rem; margin-top: .2rem; }
    .score {
        color: var(--green); font: 500 1rem 'DM Mono', monospace;
        background: var(--mint); padding: .38rem .52rem; border-radius: 5px; white-space: nowrap;
    }
    .skill-pill {
        display: inline-block; border: 1px solid #b9d5c7; color: var(--green);
        border-radius: 999px; padding: .22rem .55rem; margin: .65rem .25rem 0 0;
        font: 500 .7rem 'DM Mono', monospace;
    }
    .reasoning { color: #33443d !important; font-size: .88rem; line-height: 1.55; margin-top: .7rem; }
    .evidence {
        border-left: 3px solid var(--gold); padding-left: .75rem; color: #58645f;
        font-size: .8rem; line-height: 1.5; margin-top: .7rem;
    }
    .candidate-card .candidate-name,
    .candidate-card .candidate-meta,
    .candidate-card .reasoning,
    .candidate-card .evidence,
    .candidate-card .skill-pill {
        opacity: 1 !important;
    }
    .metric-box {
        background: rgba(255,255,255,.75); border: 1px solid var(--line);
        border-radius: 8px; padding: .85rem 1rem;
    }
    .metric-value { font: 500 1.35rem 'DM Mono', monospace; color: var(--green); }
    .metric-label { color: var(--muted); font-size: .72rem; margin-top: .15rem; }
    .empty-state {
        border: 1px dashed #b9c9c0; border-radius: 8px; padding: 2.5rem;
        text-align: center; color: var(--muted); background: rgba(255,255,255,.35);
    }
    .stChatMessage { background: rgba(255,255,255,.58); border: 1px solid var(--line); }
    .stButton > button { border-radius: 6px; font-weight: 700; }
    div[data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 8px; background: rgba(255,255,255,.55); }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
    [data-testid="stSidebar"] label {
        color: #edf8f1 !important;
    }
    [data-testid="stSidebar"] .section-label,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: #b9d2c5 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def initialise_session() -> None:
    """Create the values that must survive Streamlit reruns."""
    if "agent_app" not in st.session_state:
        st.session_state.agent_app = build_workflow()
    if "agent_state" not in st.session_state:
        st.session_state.agent_state = {
            "conversation_history": [],
            "job_requirement_understanding": {},
        }
    if "messages" not in st.session_state:
        st.session_state.messages = []


def reset_session() -> None:
    """Start a fresh matching conversation."""
    st.session_state.agent_state = {
        "conversation_history": [],
        "job_requirement_understanding": {},
    }
    st.session_state.messages = []


def candidate_label(candidate: dict) -> str:
    name = candidate.get("candidate_name", "Unknown candidate")
    score = candidate.get("match_score", 0)
    return f"{name}  |  {score:g}/100"


def render_candidate(candidate: dict, position: int) -> None:
    """Render one ranked candidate as an evidence card."""
    name = candidate.get("candidate_name", "Unknown candidate")
    score = candidate.get("match_score", 0)
    experience = candidate.get("experience_years")
    skills = candidate.get("matched_skills", [])
    reasoning = candidate.get("reasoning", "No explanation available.")
    excerpts = candidate.get("relevant_excerpts", [])

    experience_text = "Experience not available"
    if experience is not None:
        experience_text = f"{experience:g} years experience"

    skill_html = "".join(
        f'<span class="skill-pill">{skill}</span>' for skill in skills
    )
    evidence_html = ""
    if excerpts:
        evidence_html = f'<div class="evidence">{excerpts[0]}</div>'

    st.markdown(
        f"""
        <div class="candidate-card">
            <div class="candidate-top">
                <div>
                    <div class="candidate-name">{position:02d} &nbsp; {name}</div>
                    <div class="candidate-meta">{experience_text}</div>
                </div>
                <div class="score">{score:g} / 100</div>
            </div>
            <div>{skill_html or '<span class="candidate-meta">No matched skills listed</span>'}</div>
            <div class="reasoning">{reasoning}</div>
            {evidence_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            '<div class="brand-mark"><span class="brand-dot">TA</span> Talent Atlas</div>',
            unsafe_allow_html=True,
        )
        st.caption("RAG-powered resume intelligence")
        st.divider()

        if st.button("Start new search", use_container_width=True):
            reset_session()
            st.rerun()

        requirements = st.session_state.agent_state.get(
            "job_requirement_understanding", {}
        )
        st.markdown('<div class="section-label">Current brief</div>', unsafe_allow_html=True)
        job_description = requirements.get("job_description")
        if job_description:
            st.caption(job_description)
            must_have = requirements.get("must_have", [])
            nice_to_have = requirements.get("nice_to_have", [])
            if must_have:
                st.markdown("**Must-have**")
                st.write(", ".join(must_have))
            if nice_to_have:
                st.markdown("**Nice-to-have**")
                st.write(", ".join(nice_to_have))
            years = requirements.get("minimum_experience_years")
            if years is not None:
                st.markdown(f"**Minimum experience:** {years:g} years")
        else:
            st.caption("Your job requirements will appear here after the first search.")

        st.divider()
        st.markdown('<div class="section-label">Try a query</div>', unsafe_allow_html=True)
        st.caption("You can refine the search in the chat without starting over.")


def render_header() -> None:
    st.markdown('<div class="eyebrow">Candidate intelligence / 01</div>', unsafe_allow_html=True)
    st.markdown("<h1>Find the people<br>behind the keywords.</h1>", unsafe_allow_html=True)
    st.markdown(
        "<div class='hero-copy'>Search your resume library with a natural-language brief. "
        "Talent Atlas combines semantic retrieval, keyword matching, and LLM reasoning "
        "so every shortlist comes with evidence.</div>",
        unsafe_allow_html=True,
    )


def render_chat() -> None:
    st.markdown('<div class="section-label">Conversation</div>', unsafe_allow_html=True)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input(
        "Try: Find React candidates with 4+ years of experience..."
    )
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        progress_messages = {
            "decide_action": "Choosing the best workflow for your request",
            "parse_jd": "Reading the job description with the LLM",
            "extract_requirements": "Extracting must-have and nice-to-have skills",
            "search_resumes": "Searching the resume collection",
            "rank_candidates": "Ranking candidates and preparing screening rounds",
            "generate_report": "Writing the candidate report",
            "human_feedback_loop": "Saving this conversation for follow-up questions",
        }

        with st.status("Working through your request...", expanded=True) as status:
            def show_progress(node_name: str) -> None:
                message = progress_messages.get(node_name, node_name)
                status.write(f"Done: {message}")

            try:
                result = run_query(
                    st.session_state.agent_app,
                    prompt,
                    st.session_state.agent_state,
                    progress_callback=show_progress,
                )
                st.session_state.agent_state = result
                answer = result.get("report", "No report was generated.")
                status.update(
                    label="Search complete",
                    state="complete",
                    expanded=False,
                )
            except Exception as error:
                answer = "The search could not be completed."
                status.update(
                    label="Search stopped with an error",
                    state="error",
                    expanded=True,
                )
                st.error(
                    "The search could not be completed. Check the API key, model "
                    f"dependencies, and resume database. Details: {error}"
                )
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()


def render_results() -> None:
    state = st.session_state.agent_state
    candidates = state.get("shortlisted_candidates", [])
    if not candidates:
        st.markdown(
            '<div class="empty-state"><strong>Your shortlist will appear here.</strong><br>'
            "Ask for candidates above to begin a search.</div>",
            unsafe_allow_html=True,
        )
        return

    st.markdown('<div class="section-label">Shortlist</div>', unsafe_allow_html=True)
    metrics = st.columns(4)
    with metrics[0]:
        st.markdown(
            f'<div class="metric-box"><div class="metric-value">{len(candidates)}</div>'
            '<div class="metric-label">candidates found</div></div>',
            unsafe_allow_html=True,
        )
    with metrics[1]:
        top_score = candidates[0].get("match_score", 0)
        st.markdown(
            f'<div class="metric-box"><div class="metric-value">{top_score:g}</div>'
            '<div class="metric-label">top match score</div></div>',
            unsafe_allow_html=True,
        )
    with metrics[2]:
        required = state.get("job_requirement_understanding", {}).get("must_have", [])
        st.markdown(
            f'<div class="metric-box"><div class="metric-value">{len(required)}</div>'
            '<div class="metric-label">must-have skills</div></div>',
            unsafe_allow_html=True,
        )
    with metrics[3]:
        recommendations = state.get("screening_rounds", {}).get("final_recommendations", [])
        hire_count = sum(item.get("recommendation") == "hire" for item in recommendations)
        st.markdown(
            f'<div class="metric-box"><div class="metric-value">{hire_count}</div>'
            '<div class="metric-label">top recommendations</div></div>',
            unsafe_allow_html=True,
        )

    st.write("")
    for position, candidate in enumerate(candidates, start=1):
        render_candidate(candidate, position)


def render_tools() -> None:
    state = st.session_state.agent_state
    candidates = state.get("shortlisted_candidates", [])
    if not candidates:
        return

    st.markdown('<div class="section-label">Review tools</div>', unsafe_allow_html=True)
    compare_tab, interview_tab, screening_tab = st.tabs(
        ["Compare candidates", "Interview questions", "Screening rounds"]
    )

    with compare_tab:
        labels = [candidate_label(candidate) for candidate in candidates]
        selected_labels = st.multiselect(
            "Select candidates to compare",
            labels,
            default=labels[:2],
            max_selections=5,
        )
        selected_candidates = [
            candidate for candidate in candidates if candidate_label(candidate) in selected_labels
        ]
        if st.button("Build comparison", type="primary"):
            selected_ids = [
                candidate.get("candidate_name", "") for candidate in selected_candidates
            ]
            comparison = compare_candidates(selected_ids, candidates)
            st.markdown(comparison)

    with interview_tab:
        labels = [candidate_label(candidate) for candidate in candidates]
        selected_label = st.selectbox("Choose a candidate", labels)
        selected_candidate = next(
            candidate for candidate in candidates if candidate_label(candidate) == selected_label
        )
        if st.button("Generate tailored questions", type="primary"):
            required_skills = state.get("job_requirement_understanding", {}).get(
                "must_have", []
            )
            questions = generate_interview_questions(
                selected_candidate.get("candidate_name", ""),
                candidates,
                required_skills,
            )
            for number, question in enumerate(questions, start=1):
                st.markdown(f"**{number}.** {question}")

    with screening_tab:
        rounds = state.get("screening_rounds", {})
        round_one, round_two, round_three = st.columns(3)
        with round_one:
            st.markdown("**01 / Initial**")
            st.caption(f"{len(rounds.get('initial', []))} candidates")
        with round_two:
            st.markdown("**02 / Deep analysis**")
            st.caption(f"{len(rounds.get('deep_analysis', []))} candidates")
        with round_three:
            st.markdown("**03 / Recommendation**")
            st.caption(f"{len(rounds.get('final_recommendations', []))} reviewed")

        for item in rounds.get("deep_analysis", []):
            with st.expander(item.get("candidate_id", "Candidate")):
                st.write("**Strengths:**", ", ".join(item.get("strengths", [])) or "None listed")
                st.write("**Gaps:**", ", ".join(item.get("gaps", [])) or "None listed")
                for evidence in item.get("evidence", []):
                    st.caption(evidence)


def main() -> None:
    initialise_session()
    render_sidebar()
    render_header()
    render_chat()
    render_results()
    render_tools()


if __name__ == "__main__":
    main()
