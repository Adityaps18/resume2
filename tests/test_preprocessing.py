"""Unit tests for text preprocessing and normalization."""

import unittest
from src.preprocessing import clean_text, extract_keywords_list


class TestPreprocessing(unittest.TestCase):

    def test_clean_text_removes_urls_and_emails(self):
        sample = "Reach me at test.user@example.com or visit https://github.com/myprofile."
        cleaned = clean_text(sample)
        self.assertNotIn("test.user@example.com", cleaned)
        self.assertNotIn("https://github.com/myprofile", cleaned)
        self.assertIn("reach me at", cleaned)

    def test_clean_text_removes_phone_numbers(self):
        sample = "Call me at +1 (555) 123-4567 or 9876543210 for inquiries."
        cleaned = clean_text(sample)
        self.assertNotIn("9876543210", cleaned)
        self.assertNotIn("555", cleaned)
        self.assertIn("call me at", cleaned)

    def test_clean_text_preserves_tech_tokens(self):
        sample = "Expert in C++, C#, .NET core, and Node.js backend development."
        cleaned = clean_text(sample)
        self.assertIn("c++", cleaned)
        self.assertIn("c#", cleaned)
        self.assertIn(".net", cleaned)
        self.assertIn("node.js", cleaned)

    def test_clean_text_handles_empty_and_none(self):
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text(None), "")
        self.assertEqual(clean_text("   \n\t  "), "")

    def test_extract_keywords_list(self):
        sample = "Python, SQL, Python, Docker and AWS."
        keywords = extract_keywords_list(sample)
        self.assertIn("python", keywords)
        self.assertIn("sql", keywords)
        self.assertIn("docker", keywords)
        self.assertIn("aws", keywords)
        self.assertEqual(len(keywords), len(set(keywords)))


if __name__ == "__main__":
    unittest.main()
