"""Unit tests for skill matching and semantic similarity engine."""

import unittest
from src.matcher import match_resume_to_job, calculate_semantic_similarity, synthesize_hiring_decision


class TestMatcher(unittest.TestCase):

    def test_match_resume_exact_and_missing_skills(self):
        resume = "Data Scientist with experience in Python, SQL, Machine Learning, and Docker with 4 years of experience."
        jd = "Seeking a Data Scientist with 3+ years experience proficient in Python, SQL, Machine Learning, and Tableau."

        result = match_resume_to_job(resume, jd)
        self.assertIn("data science", result["matched_skills"])
        self.assertIn("python", result["matched_skills"])
        self.assertIn("sql", result["matched_skills"])
        self.assertIn("machine learning", result["matched_skills"])
        self.assertIn("tableau", result["missing_skills"])
        self.assertIn("docker", result["extra_skills"])

        # 4 out of 5 skills matched = 80.0%
        self.assertEqual(result["skill_score"], 80.0)
        self.assertGreater(result["composite_score"], 50.0)
        self.assertEqual(result["candidate_experience"], 4)
        self.assertEqual(result["required_experience"], 3)
        self.assertEqual(result["experience_status"], "Exceeded")


    def test_calculate_semantic_similarity(self):
        text1 = "Senior Machine Learning Engineer designing deep neural networks with PyTorch and Python."
        text2 = "Looking for an ML engineer with PyTorch, Python and deep learning background."
        sim = calculate_semantic_similarity(text1, text2)
        self.assertGreater(sim, 10.0)

    def test_empty_job_description_handling(self):
        resume = "Python, Java, AWS developer."
        jd = ""
        result = match_resume_to_job(resume, jd)
        self.assertEqual(result["matched_skills"], [])
        self.assertEqual(result["missing_skills"], [])
        self.assertEqual(result["skill_score"], 100.0)

    def test_synthesize_decision_rejects_low_match(self):
        low_match_result = {
            "composite_score": 11.0,
            "skill_score": 0.0,
            "total_required_skills": 4,
            "missing_skills": ["python", "sql", "machine learning", "tableau"]
        }
        res = synthesize_hiring_decision(
            match_result=low_match_result,
            has_job_description=True
        )
        self.assertEqual(res["category"], "Reject")
        self.assertIn("Reject", res["decision"])
        self.assertIn("mismatch", res["decision"].lower())

    def test_synthesize_decision_moderate_match(self):
        mod_match_result = {
            "composite_score": 55.0,
            "skill_score": 50.0,
            "total_required_skills": 4,
            "missing_skills": ["tableau", "docker"]
        }
        res = synthesize_hiring_decision(
            match_result=mod_match_result,
            has_job_description=True
        )
        self.assertEqual(res["category"], "Review")
        self.assertIn("Review", res["decision"])

    def test_synthesize_decision_strong_match(self):
        high_match_result = {
            "composite_score": 88.0,
            "skill_score": 100.0,
            "total_required_skills": 4,
            "missing_skills": []
        }
        res = synthesize_hiring_decision(
            match_result=high_match_result,
            has_job_description=True
        )
        self.assertEqual(res["category"], "Hire")
        self.assertIn("Hire", res["decision"])


if __name__ == "__main__":
    unittest.main()

