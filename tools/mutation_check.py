"""Mutation check: break the code on purpose, one change at a time, and confirm that the tests turn red.

Each mutant in tools/mutants.json replaces one exact piece of text in one file. Everything runs in a temporary
copy of the project, so the working tree is never touched. A mutant is caught when its test lane fails.

Usage: python tools/mutation_check.py              all lanes (about 8 minutes on the development machine)
       python tools/mutation_check.py --lane fast  only the mutants checked by the fast tests
"""
import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
MUTANTS = ROOT / "tools" / "mutants.json"
LANES = {
    "fast": ["-m", "unittest", "discover", "-s", "tests", "-q"],
    "notebook": ["-m", "unittest", "slow_tests.test_notebook.NotebookBehaviour", "slow_tests.test_notebook.ColabBootstrap", "-q"],
}
SKIP = shutil.ignore_patterns(".git", "__pycache__", ".ipynb_checkpoints")


def stored_text(path: pathlib.Path, text: str) -> str:
    """Notebook sources live inside JSON strings, so text aimed at a notebook is matched in its escaped form."""
    return json.dumps(text, ensure_ascii=False)[1:-1] if path.suffix == ".ipynb" else text


def load_mutants() -> list[dict]:
    return json.loads(MUTANTS.read_text(encoding="utf-8"))


def occurrences(root: pathlib.Path, mutant: dict) -> int:
    """How many times the mutant's target text appears in its file; exactly one is required."""
    path = root / mutant["file"]
    # Bytes, not read_text: the count must see the same line endings that run_mutant replaces in.
    return path.read_bytes().decode("utf-8").count(stored_text(path, mutant["old"]))


def run_mutant(copy: pathlib.Path, mutant: dict) -> str:
    if occurrences(copy, mutant) != 1:
        return "INVALID"
    path = copy / mutant["file"]
    original = path.read_bytes()
    text = original.decode("utf-8")
    old, new = stored_text(path, mutant["old"]), stored_text(path, mutant["new"])
    try:
        path.write_bytes(text.replace(old, new).encode("utf-8"))
        result = subprocess.run([sys.executable, "-B", *LANES[mutant["lane"]]], cwd=copy, capture_output=True, timeout=900)
        return "CAUGHT" if result.returncode else "MISSED"
    finally:
        path.write_bytes(original)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lane", choices=sorted(LANES))
    args = parser.parse_args()
    mutants = [m for m in load_mutants() if args.lane in (None, m["lane"])]
    statuses = []
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        copy = pathlib.Path(folder) / ROOT.name
        shutil.copytree(ROOT, copy, ignore=SKIP)
        for number, mutant in enumerate(mutants, start=1):
            statuses.append(run_mutant(copy, mutant))
            print(f"[{number}/{len(mutants)}] {statuses[-1]:<7} {mutant['id']}", flush=True)
    caught = statuses.count("CAUGHT")
    print(f"{caught} of {len(mutants)} mutants caught")
    sys.exit(0 if caught == len(mutants) else 1)


if __name__ == "__main__":
    main()
