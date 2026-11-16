"""Every text file in the repository is free of control characters and of the long dash characters."""
import json
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
TEXT_SUFFIXES = {".py", ".md", ".txt", ".csv", ".json", ".ipynb", ".gitignore"}
FORBIDDEN = {chr(0x2014): "em dash", chr(0x2013): "en dash"}


def problems_in(text: str) -> list[str]:
    found = [f"control character {hex(ord(c))}" for c in set(text) if ord(c) < 32 and c not in "\n\r\t"]
    found += [name for char, name in FORBIDDEN.items() if char in text]
    return sorted(found)


def text_files() -> list[pathlib.Path]:
    return sorted(
        p for p in ROOT.rglob("*")
        if p.is_file() and (p.suffix in TEXT_SUFFIXES or p.name in TEXT_SUFFIXES) and ".git" not in p.parts and "__pycache__" not in p.parts
    )


def notebook_text(path: pathlib.Path) -> str:
    """Cell sources and text outputs; image outputs are base64 and skipped."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    parts = []
    for cell in notebook["cells"]:
        parts.append("".join(cell["source"]))
        for output in cell.get("outputs", []):
            parts.append("".join(output.get("text", "")))
            parts.append("".join(output.get("data", {}).get("text/plain", "")))
    return "\n".join(parts)


class TextHygiene(unittest.TestCase):
    def test_no_file_contains_control_characters_or_long_dashes(self):
        files = text_files()
        self.assertGreaterEqual(len(files), 15)
        report = {}
        for path in files:
            text = notebook_text(path) if path.suffix == ".ipynb" else path.read_text(encoding="utf-8")
            if problems_in(text):
                report[str(path.relative_to(ROOT))] = problems_in(text)
        self.assertEqual(report, {})

    def test_checker_flags_each_kind_of_problem(self):
        self.assertEqual(problems_in("temiz metin\n\tsatır\r\n"), [])
        self.assertEqual(problems_in("a" + chr(8) + "b"), ["control character 0x8"])
        self.assertEqual(problems_in("bir" + chr(0x2014) + "iki"), ["em dash"])
        self.assertEqual(problems_in("bir" + chr(0x2013) + "iki"), ["en dash"])


if __name__ == "__main__":
    unittest.main()
