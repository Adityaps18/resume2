"""Unit tests for automated resume parsing and entity extraction."""

import unittest
from src.parser import (
    extract_skills,
    estimate_experience_years,
    extract_education,
    extract_certifications,
    estimate_project_count,
    parse_resume_full,
    extract_required_experience_from_jd,
    extract_required_education_from_jd
)


class TestParser(unittest.TestCase):

    def test_extract_skills_boundary_safety(self):
        # JavaScript should NOT extract Java
        text_js = "Experienced frontend engineer skilled in JavaScript, HTML5, and CSS3."
        skills_js = extract_skills(text_js)
        self.assertIn("javascript", skills_js)
        self.assertNotIn("java", skills_js)
        self.assertIn("html", skills_js)
        self.assertIn("css", skills_js)

    def test_extract_skills_multiword_and_aliases(self):
        text = "Implemented Deep Learning models using PyTorch, ML pipelines, Next.js, and K8s on AWS."
        skills = extract_skills(text)
        self.assertIn("deep learning", skills)
        self.assertIn("pytorch", skills)
        self.assertIn("machine learning", skills)  # ML alias
        self.assertIn("kubernetes", skills)        # K8s alias
        self.assertIn("next.js", skills)
        self.assertIn("aws", skills)

    def test_estimate_experience_explicit(self):
        text = "Senior Software Engineer with 6+ years of experience in distributed systems."
        exp = estimate_experience_years(text)
        self.assertEqual(exp, 6)

    def test_estimate_experience_date_ranges(self):
        text = "Work History:\nSoftware Engineer at TechCorp (2018 - 2022)\nJunior Developer (2016 - 2018)"
        exp = estimate_experience_years(text)
        self.assertEqual(exp, 6)

    def test_extract_education(self):
        self.assertEqual(extract_education("Completed B.Tech in Computer Science and Engineering"), "B.Tech")
        self.assertEqual(extract_education("Holds a Master of Science (M.S.) in Artificial Intelligence"), "M.Tech")
        self.assertEqual(extract_education("Completed MBA in Operations & Analytics"), "MBA")
        self.assertEqual(extract_education("Candidate holds a Ph.D in Neural Networks"), "PhD")

    def test_extract_certifications(self):
        text = "Holds AWS Certified Solutions Architect Associate."
        self.assertEqual(extract_certifications(text), "AWS Certified")

    def test_estimate_experience_fresher_education_dates_not_counted(self):
        # A 4-year B.Tech degree date range (2020 - 2024) MUST NOT be counted as 4 years work experience!
        fresher_text = """
        Aditya Pratap Singh
        EDUCATION:
        Bachelor of Technology in Computer Science (2020 - 2024)
        Class XII CBSE Board (2018 - 2020)
        
        SKILLS:
        Python, Data Science, Machine Learning, SQL
        
        PROJECTS:
        Project 1: Machine Learning Model (2023 - 2024)
        """
        exp = estimate_experience_years(fresher_text)
        self.assertEqual(exp, 0)

    def test_extract_required_experience_from_jd(self):
        jd1 = "Looking for a candidate with 3+ years of experience in Python and SQL."
        self.assertEqual(extract_required_experience_from_jd(jd1), 3)

        jd2 = "Minimum 5 years of industry experience required."
        self.assertEqual(extract_required_experience_from_jd(jd2), 5)

        jd3 = "Data Science – Fresher: Responsible for building software applications..."
        self.assertEqual(extract_required_experience_from_jd(jd3), 0)

        jd4 = "Looking for a developer with React and Node.js skills."
        self.assertIsNone(extract_required_experience_from_jd(jd4))


    def test_extract_required_education_from_jd(self):
        jd1 = "Requires a Bachelor's degree in Computer Science or related field."
        self.assertIn("Bachelor's", extract_required_education_from_jd(jd1))

        jd2 = "Master's degree or Ph.D required."
        self.assertEqual(extract_required_education_from_jd(jd2), "PhD")

    def test_parse_resume_full(self):
        sample_resume = """
        Jane Doe
        Email: jane.doe@example.com | Phone: 9876543210
        
        Summary:
        Data Scientist with 4 years of experience building Machine Learning and Deep Learning solutions in Python and SQL.
        
        Education:
        Bachelor of Technology in Computer Science (2018 - 2022)
        
        Certifications:
        Google ML Engineer Certified
        
        Projects:
        Project 1: Real-time Object Detection with PyTorch
        Project 2: Customer Churn Prediction using Scikit-Learn
        Project 3: Automated ETL Pipeline using Apache Spark and Docker
        """
        profile = parse_resume_full(sample_resume)
        self.assertIn("python", profile["skills"])
        self.assertIn("machine learning", profile["skills"])
        self.assertIn("deep learning", profile["skills"])
        self.assertIn("pytorch", profile["skills"])
        self.assertEqual(profile["experience_years"], 4)
        self.assertEqual(profile["education"], "B.Tech")
        self.assertIn("Google", profile["certification"])
        self.assertGreaterEqual(profile["projects_count"], 3)


if __name__ == "__main__":
    unittest.main()

