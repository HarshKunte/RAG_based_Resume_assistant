import unittest
from unittest.mock import patch

from matching_agent import build_workflow, decide_action, run_query
from job_matcher import _normalise_skill


CANDIDATES = [
    {
        "candidate_id": "john_doe",
        "candidate_name": "John Doe",
        "match_score": 92,
        "matched_skills": ["React", "Python"],
        "qualified": True,
        "reasoning": "Strong React and Python evidence.",
        "relevant_excerpts": ["Built React applications for five years."],
    },
    {
        "candidate_id": "jane_smith",
        "candidate_name": "Jane Smith",
        "match_score": 71,
        "matched_skills": ["React"],
        "qualified": True,
        "reasoning": "React evidence, but less relevant experience.",
        "relevant_excerpts": ["Maintained a React frontend."],
    },
]


def fake_extract(_jd):
    return {"skills": {"must_have": ["React"], "nice_to_have": ["Python"]}, "minimum_experience_years": 3}


def fake_search(_query, top_k=10):
    return {"top_matches": CANDIDATES[:top_k]}


class MatchingAgentConversationTests(unittest.TestCase):
    def setUp(self):
        self.matcher = patch("matching_agent._matcher", return_value=(fake_extract, fake_search))
        self.matcher.start()
        self.app = build_workflow()

    def tearDown(self):
        self.matcher.stop()

    def ask(self, query, state=None):
        return run_query(self.app, query, state)

    def test_initial_candidate_search(self):
        result = self.ask("Find candidates with React and 3+ years experience")
        self.assertIn("John Doe", result["report"])
        self.assertEqual(len(result["screening_rounds"]["initial"]), 2)

    def test_compare_top_three(self):
        result = self.ask("Compare the top 3 matches side by side")
        self.assertIn("Candidate | Score", result["report"])

    def test_compare_single_word_candidate_names(self):
        single_name_candidates = [
            {
                "candidate_id": "harsh",
                "candidate_name": "Harsh",
                "match_score": 92,
                "matched_skills": ["React"],
                "qualified": True,
                "reasoning": "Strong React evidence.",
                "relevant_excerpts": ["Built React applications for five years."],
            },
            {
                "candidate_id": "grace",
                "candidate_name": "Grace",
                "match_score": 88,
                "matched_skills": ["React"],
                "qualified": True,
                "reasoning": "Solid React experience.",
                "relevant_excerpts": ["Delivered React interfaces."],
            },
        ]
        state = {
            "query": "Compare Harsh and Grace",
            "conversation_history": [],
            "job_requirement_understanding": {"must_have": ["React"]},
            "shortlisted_candidates": single_name_candidates,
            "previous_shortlist": [],
        }
        result = self.ask("Compare Harsh and Grace", state)
        self.assertIn("Candidate | Score", result["report"])
        self.assertIn("Harsh", result["report"])

    def test_decision_node_uses_existing_shortlist(self):
        self.assertEqual(
            decide_action({"query": "Compare the top 2", "shortlisted_candidates": CANDIDATES})[
                "next_step"
            ],
            "generate_report",
        )

    def test_follow_up_comparison_skips_retrieval(self):
        first_result = self.ask("Find React candidates")
        self.matcher.side_effect = AssertionError("retrieval should be skipped")
        second_result = self.ask("Compare the top 2 matches", first_result)
        self.assertIn("Candidate | Score", second_result["report"])

    def test_explain_ranking(self):
        result = self.ask("Why did John rank higher than Jane?")
        self.assertIn("ranked higher", result["report"])

    def test_interview_questions(self):
        result = self.ask("Generate interview questions for John Doe")
        self.assertIn("production work", result["report"])

    def test_requirement_refinement_keeps_history(self):
        first = self.ask("Find React candidates")
        second = self.ask("Now require Python too", first)
        self.assertGreaterEqual(len(second["conversation_history"]), 4)
        self.assertIn("job_description", second["job_requirement_understanding"])

    def test_typescript_aliases_normalise_to_same_skill(self):
        self.assertEqual(_normalise_skill("TypeScript"), "typescript")
        self.assertEqual(_normalise_skill("typescript"), "typescript")

    def test_skill_punctuation_is_normalised_generically(self):
        self.assertEqual(_normalise_skill("React.js"), "react js")
        self.assertEqual(_normalise_skill("react-js"), "react js")


if __name__ == "__main__":
    unittest.main()
