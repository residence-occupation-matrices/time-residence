import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class DocumentationTests(unittest.TestCase):
    def test_documentation_has_no_control_characters(self):
        paths = sorted(REPOSITORY_ROOT.rglob("*.md")) + sorted(REPOSITORY_ROOT.rglob("*.org"))
        for path in paths:
            with self.subTest(path=path.relative_to(REPOSITORY_ROOT)):
                text = path.read_text(encoding="utf-8")
                invalid = [
                    (index, ord(char))
                    for index, char in enumerate(text)
                    if ord(char) < 32 and char not in "\n\r"
                ]
                self.assertEqual(invalid, [])

    def test_github_math_uses_supported_display_delimiters(self):
        for relative_path in ("docs/DATA_CONTRACT.md", "docs/METHODS.md"):
            with self.subTest(path=relative_path):
                text = (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")
                self.assertNotIn(r"\[", text)
                self.assertNotIn(r"\]", text)
                self.assertEqual(text.count("$$") % 2, 0)


if __name__ == "__main__":
    unittest.main()
