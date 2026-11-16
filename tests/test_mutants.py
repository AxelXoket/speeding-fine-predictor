"""The mutation list stays usable: every mutant still points at text that exists exactly once, and the README count is true."""
import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("mutation_check", ROOT / "tools" / "mutation_check.py")
mutation_check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mutation_check)
MUTANTS = mutation_check.load_mutants()


class MutantListIsCurrent(unittest.TestCase):
    def test_every_mutant_targets_text_that_exists_exactly_once(self):
        self.assertGreaterEqual(len(MUTANTS), 60)
        counts = {m["id"]: mutation_check.occurrences(ROOT, m) for m in MUTANTS}
        self.assertEqual({k: v for k, v in counts.items() if v != 1}, {})

    def test_mutants_are_well_formed(self):
        # The short label in front ('F9') is how a mutant is named in a run's output, so it must be unique too.
        labels = [m["id"].split(" ", 1)[0] for m in MUTANTS]
        self.assertEqual(sorted(label for label in set(labels) if labels.count(label) > 1), [])
        for m in MUTANTS:
            with self.subTest(mutant=m["id"]):
                self.assertRegex(m["id"], r"^[A-Z]\d+ \S")
                # Git stores LF and a Windows checkout has CRLF, so a target with a line break matches on one of them only.
                # Notebook targets are exempt: their line breaks are '\n' escapes inside JSON strings.
                if not m["file"].endswith(".ipynb"):
                    self.assertNotIn("\n", m["old"] + m["new"])
                self.assertIn(m["lane"], mutation_check.LANES)
                self.assertNotEqual(m["old"], m["new"])
        self.assertEqual({m["lane"] for m in MUTANTS}, set(mutation_check.LANES))

    def test_readme_states_the_real_number(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn(f"tek tek bozan {len(MUTANTS)} değişikliğin hepsini", readme)

    def test_a_stale_target_is_reported(self):
        stale = {"id": "x", "lane": "fast", "file": "README.md", "old": "bu metin README'de hiç yok", "new": "y"}
        self.assertEqual(mutation_check.occurrences(ROOT, stale), 0)
        # Notebook text is matched in its JSON-escaped form, quotes included.
        quoted = next(m for m in MUTANTS if m["file"].endswith(".ipynb") and '"' in m["old"])
        self.assertEqual(mutation_check.occurrences(ROOT, quoted), 1)


if __name__ == "__main__":
    unittest.main()
