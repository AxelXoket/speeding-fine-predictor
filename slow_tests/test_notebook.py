"""Execute the notebook in a scratch copy of the project and check what its code actually computes.

Slow lane (a kernel start per test). Run with: python -m unittest discover -s slow_tests
"""
import importlib.util
import json
import pathlib
import shutil
import sys
import tempfile
import unittest

import nbformat
from nbclient import NotebookClient

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from speeding_fine import dataset, rules  # noqa: E402

_spec = importlib.util.spec_from_file_location("build_notebook", ROOT / "tools" / "build_notebook.py")
build_notebook = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_notebook)

PROBE_MARK = "PROBE_JSON:"

# Runs after the notebook in the same kernel and reports what the notebook's objects really do.
PROBE = r'''
import builtins, contextlib, io, json
from speeding_fine import rules

def run_with_answers(answers):
    feed = iter(answers)
    original = builtins.input
    builtins.input = lambda prompt="": next(feed)
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            run_fine_assistant()
    finally:
        builtins.input = original
    return buffer.getvalue()

def run_without_keyboard(**options):
    original = builtins.input
    def refuse(prompt=""):
        raise EOFError
    builtins.input = refuse
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            run_fine_assistant(**options)
    finally:
        builtins.input = original
    return buffer.getvalue()

def law_fine_at(urban, excess):
    vehicle, road = ("otomobil", "yerlesim_yeri") if urban else ("otomobil", "otoyol")
    return rules.penalty_for(vehicle, road, rules.speed_limit(vehicle, road) + excess).fine_tl

def model_fine_at(urban, excess):
    return int(model.predict(pd.DataFrame([[excess, urban]], columns=ENGINEERED))[0])

def fine_start_recount(urban):
    """[largest unfined excess, smallest fined excess] on one side, from the raw columns rather than asim_km."""
    rows = zip(clean["olculen_hiz"], clean["hiz_siniri"], clean["yol_tipi"], clean["ceza_tutari_tl"])
    pairs = [(speed - limit, fine) for speed, limit, road, fine in rows if (road == rules.URBAN_ROAD) == urban]
    return [int(max(e for e, fine in pairs if fine == 0)), int(min(e for e, fine in pairs if fine > 0))]

boundaries = {1: [t.min_excess for t in rules.URBAN_TIERS], 0: [t.min_excess for t in rules.RURAL_TIERS]}
mid_tier = {1: [-10, 3, 8, 13, 18, 23, 30, 40, 50, 60, 80], 0: [-10, 5, 13, 18, 23, 28, 35, 45, 55, 65, 85]}
mid_tier_checks = []
for urban, road, vehicle in [(1, "yerlesim_yeri", "otomobil"), (0, "otoyol", "kamyon"), (0, "bolunmus_yol", "otobus")]:
    for excess in mid_tier[urban]:
        speed = rules.speed_limit(vehicle, road) + excess
        predicted = predict_fine(vehicle, road, speed)
        law = rules.penalty_for(vehicle, road, speed)
        mid_tier_checks.append([predicted["fine_tl"] == law.fine_tl, predicted["suspension_days"] == law.suspension_days])

labels_match_law = all(
    (row.ceza_tutari_tl, row.ehliyet_geri_alma_gun)
    == (rules.penalty_for(row.arac_tipi, row.yol_tipi, int(row.olculen_hiz)).fine_tl,
        rules.penalty_for(row.arac_tipi, row.yol_tipi, int(row.olculen_hiz)).suspension_days)
    for row in clean.itertuples(index=False)
)
law_fines = [rules.penalty_for(v, r, int(s)).fine_tl for v, r, s in zip(grid["arac_tipi"], grid["yol_tipi"], grid["olculen_hiz"])]
recount = {
    "Model A (ham)": sum(p == f for p, f in zip(raw_model.predict(raw_matrix(grid, X_raw_train.columns)), law_fines)) / len(law_fines),
    "Model B (aşım)": sum(p == f for p, f in zip(model.predict(grid[ENGINEERED]), law_fines)) / len(law_fines),
}
seen_pairs = set(zip(X_eng_train["asim_km"], X_eng_train["yerlesim_ici"]))
grid_unseen = [pair not in seen_pairs for pair in zip(grid["asim_km"], grid["yerlesim_ici"])]
grid_hits = [p == f for p, f in zip(model.predict(grid[ENGINEERED]), law_fines)]
unseen_hits = [hit for hit, new in zip(grid_hits, grid_unseen) if new]
seen_hits = [hit for hit, new in zip(grid_hits, grid_unseen) if not new]
boundary_recount = sum(
    all(model_fine_at(urban, start + step) == law_fine_at(urban, start + step) for step in (-1, 0))
    for urban, tiers in ((1, rules.URBAN_TIERS), (0, rules.RURAL_TIERS))
    for start in [tier.min_excess for tier in tiers]
)
test_recount = {
    "Model A (ham)": float((raw_model.predict(X_raw_test) == y_test.to_numpy()).mean()),
    "Model B (aşım)": float((model.predict(X_eng_test) == y_test.to_numpy()).mean()),
}
report = {
    "law_recount": recount,
    "test_recount": test_recount,
    "disagreement_recount": grid_hits.count(False),
    "unseen_share_recount": sum(grid_unseen) / len(grid_unseen),
    "unseen_agreement_recount": sum(unseen_hits) / len(unseen_hits),
    "seen_agreement_recount": sum(seen_hits) / len(seen_hits),
    "boundary_recount": boundary_recount,
    "distance_recount": max(min(abs(int(e) - start) for start in boundaries[u]) for u, e in mistake_points),
    "boundary_total": len(rules.URBAN_TIERS) + len(rules.RURAL_TIERS),
    "fine_start": {str(u): [int(fine_start.loc[(u, 0), "max"]), int(fine_start.loc[(u, 1), "min"])] for u in (0, 1)},
    "fine_start_recount": {str(u): fine_start_recount(bool(u)) for u in (0, 1)},
    "limiter_rows": limiter_rows,
    "limiter_recount": sum(
        1 for v, s in zip(clean["arac_tipi"], clean["olculen_hiz"]) if v in rules.SPEED_LIMITER and s > rules.SPEED_LIMITER[v]
    ),
    "distinct_inputs": distinct_inputs,
    "distinct_recount": len(set(zip(grid["asim_km"], grid["yerlesim_ici"]))),
    "assistant_typed": run_without_keyboard(read=typed_answers(["Otomobil", "bölünmüş yol", "165"])),
    "grid_size": len(grid),
    "clean_rows": len(clean),
    "clean_nulls": int(clean.isnull().sum().sum()),
    "clean_duplicates": int(clean.duplicated().sum()),
    "vehicles": sorted(set(clean["arac_tipi"])),
    "roads": sorted(set(clean["yol_tipi"])),
    "speed_min": int(clean["olculen_hiz"].min()),
    "speed_max": int(clean["olculen_hiz"].max()),
    "unknown_weather": int((clean["hava"] == "bilinmiyor").sum()),
    "labels_match_law": labels_match_law,
    "results": {k: {m: float(v) for m, v in d.items()} for k, d in results.items()},
    "mistake_distance": [min(abs(e - b) for b in boundaries[u] + [b - 1 for b in boundaries[u]]) for u, e in mistake_points],
    "curve": [float(a) for a in curve["Ortalama"]],
    "curve_worst": [float(a) for a in curve["En kötü"]],
    "mid_tier_checks": mid_tier_checks,
    "mid_tier_warnings": sum(
        "Uyarı" in describe_prediction(v, r, rules.speed_limit(v, r) + e)
        for u, r, v in [(1, "yerlesim_yeri", "otomobil"), (0, "otoyol", "kamyon")]
        for e in mid_tier[u]
    ),
    "mistake_sentences": [
        describe_prediction("otomobil", "yerlesim_yeri" if u else "otoyol", rules.speed_limit("otomobil", "yerlesim_yeri" if u else "otoyol") + int(e))
        for u, e in mistake_points
    ],
    "mistake_laws": [
        rules.penalty_for("otomobil", "yerlesim_yeri" if u else "otoyol", rules.speed_limit("otomobil", "yerlesim_yeri" if u else "otoyol") + int(e)).fine_tl
        for u, e in mistake_points
    ],
    "example_sentences": [describe_prediction(v, r, s) for v, r, s in examples],
    "weather": {k: [float(m), int(n)] for k, (m, n) in clean.groupby("hava")["ceza_var"].agg(["mean", "size"]).iterrows()},
    "assistant_labels": run_with_answers(["Minibüs", "Yerleşim yeri içi", "77"]),
    "assistant_upper": run_with_answers(["MİNİBÜS", "OTOYOL", "³", "126"]),
    "assistant_give_up": run_with_answers(["x", "y", "z", "1", "q", "w", "e"]),
    "no_fine": describe_prediction("otomobil", "yerlesim_yeri", 54),
    "fine": describe_prediction("panelvan", "otoyol", 132),
    "suspension": describe_prediction("otomobil", "bolunmus_yol", 165),
    "assistant_retry": run_with_answers(["9", "otomobil", "0", "1", "abc", "400", "100"]),
    "assistant_names": run_with_answers(["Kamyon", "otoyol", "105"]),
    "assistant_default": run_without_keyboard(),
    "default_sentence": describe_prediction("otomobil", "sehirlerarasi_cift_yonlu", 104),
    "figures": sorted(os.listdir("figures")),
}
print("PROBE_JSON:" + json.dumps(report, ensure_ascii=False))
'''

# Replaces the clone in the first cell: records the call and copies the project instead of downloading it.
FAKE_CLONE = r'''
import shutil, subprocess
CLONE_CALLS = []
def fake_run(args, check=False):
    CLONE_CALLS.append(list(args))
    shutil.copytree(SOURCE, "speeding-fine-predictor")
subprocess.run = fake_run
'''


def copy_project(target: pathlib.Path) -> None:
    for folder in ("speeding_fine", "data"):
        shutil.copytree(ROOT / folder, target / folder, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(build_notebook.NOTEBOOK, target / build_notebook.NOTEBOOK.name)


def execute(notebook: nbformat.NotebookNode, workdir: pathlib.Path) -> nbformat.NotebookNode:
    NotebookClient(notebook, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(workdir)}}).execute()
    return notebook


def probe_report(notebook: nbformat.NotebookNode) -> dict:
    for output in notebook.cells[-1].get("outputs", []):
        for line in output.get("text", "").splitlines():
            if line.startswith(PROBE_MARK):
                return json.loads(line[len(PROBE_MARK):])
    raise AssertionError("probe printed nothing")


class CommittedNotebook(unittest.TestCase):
    def test_matches_the_builder_and_every_code_cell_ran_without_error(self):
        committed = nbformat.read(build_notebook.NOTEBOOK, as_version=4)
        built = build_notebook.build()
        self.assertEqual([(c.cell_type, c.source) for c in committed.cells], [(c.cell_type, c.source) for c in built.cells])
        code_cells = [c for c in committed.cells if c.cell_type == "code"]
        self.assertGreaterEqual(len(code_cells), 15)
        for cell in code_cells:
            self.assertIsNotNone(cell.execution_count)
            self.assertFalse([o for o in cell.outputs if o.output_type == "error"])

    def test_metrics_are_saved_before_the_cell_that_waits_for_the_keyboard(self):
        sources = [c.source for c in build_notebook.build().cells]
        metrics_cell = next(i for i, s in enumerate(sources) if "figures/metrics.json" in s)
        interactive_cell = next(i for i, s in enumerate(sources) if s.rstrip().endswith("run_fine_assistant()"))
        self.assertLess(metrics_cell, interactive_cell)
        self.assertEqual(interactive_cell, max(i for i, c in enumerate(build_notebook.build().cells) if c.cell_type == "code"))

    def test_saved_sample_session_shows_the_typed_answers(self):
        committed = nbformat.read(build_notebook.NOTEBOOK, as_version=4)
        cell = next(c for c in committed.cells if c.cell_type == "code" and c.source.startswith("# Sample session"))
        lines = "".join(o.get("text", "") for o in cell.outputs).splitlines()
        self.assertNotIn("(otomatik cevap)", "\n".join(lines))
        self.assertEqual([line.rsplit("): ", 1)[1] for line in lines[:3]], ["Otomobil", "bölünmüş yol", "165"])
        metrics = json.loads((ROOT / "figures" / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(lines[3], metrics["examples"][3])

    def test_committed_notebook_does_not_reveal_local_paths(self):
        text = build_notebook.NOTEBOOK.read_text(encoding="utf-8")
        self.assertNotIn(str(ROOT.parent), text)
        self.assertNotIn(str(ROOT.parent).replace("\\", "\\\\"), text)
        self.assertIn("Çalışma klasörü: speeding-fine-predictor", text)


class NotebookBehaviour(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        workdir = pathlib.Path(cls.folder.name)
        copy_project(workdir)
        notebook = nbformat.read(workdir / build_notebook.NOTEBOOK.name, as_version=4)
        notebook.cells.append(nbformat.v4.new_code_cell(PROBE))
        cls.report = probe_report(execute(notebook, workdir))
        cls.metrics = json.loads((workdir / "figures" / "metrics.json").read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_cleaning_removes_exactly_the_damaged_rows(self):
        r = self.report
        self.assertEqual(r["clean_rows"], dataset.DEFAULT_ROWS - dataset.MISSING_SPEED_ROWS - dataset.IMPOSSIBLE_SPEED_ROWS)
        self.assertEqual((r["clean_nulls"], r["clean_duplicates"]), (0, 0))
        self.assertEqual(r["vehicles"], sorted(rules.VEHICLE_TYPES))
        self.assertEqual(r["roads"], sorted(rules.ROAD_TYPES))
        self.assertGreaterEqual(r["speed_min"], 1)
        self.assertLessEqual(r["speed_max"], rules.MAX_PLAUSIBLE_SPEED)
        self.assertGreater(r["unknown_weather"], 0)
        self.assertTrue(r["labels_match_law"])

    def test_engineered_model_beats_raw_model_and_stays_close_to_the_law(self):
        raw, engineered = self.report["results"]["Model A (ham)"], self.report["results"]["Model B (aşım)"]
        self.assertGreater(engineered["law_agreement"], raw["law_agreement"] + 0.1)
        self.assertGreater(engineered["test_accuracy"], raw["test_accuracy"])
        self.assertGreaterEqual(engineered["test_accuracy"], 0.97)
        self.assertGreaterEqual(engineered["law_agreement"], 0.95)

    def test_reported_scores_equal_an_independent_recount_against_the_law(self):
        r = self.report
        expected_grid = len(rules.VEHICLE_TYPES) * len(rules.ROAD_TYPES) * len(range(-25, 96))
        self.assertEqual(r["grid_size"], expected_grid)
        for name in ("Model A (ham)", "Model B (aşım)"):
            with self.subTest(model=name):
                self.assertAlmostEqual(r["results"][name]["law_agreement"], r["law_recount"][name], places=12)
                self.assertAlmostEqual(r["results"][name]["test_accuracy"], r["test_recount"][name], places=12)

    def test_unseen_input_and_boundary_measures_equal_an_independent_recount(self):
        r, m = self.report, self.metrics
        self.assertEqual(m["model_b_disagreements"], r["disagreement_recount"])
        self.assertEqual(m["boundaries_total"], r["boundary_total"])
        self.assertEqual(m["boundaries_found"], r["boundary_recount"])
        self.assertEqual(m["mistake_distance_max"], r["distance_recount"])
        self.assertAlmostEqual(m["unseen_share"], r["unseen_share_recount"], places=4)
        self.assertAlmostEqual(m["unseen_agreement"], r["unseen_agreement_recount"], places=4)
        self.assertAlmostEqual(m["seen_agreement"], r["seen_agreement_recount"], places=4)
        # The share must be real: some inputs were seen in training and some were not.
        self.assertGreater(r["unseen_share_recount"], 0)
        self.assertLess(r["unseen_share_recount"], 1)

    def test_fines_start_exactly_where_the_law_starts_them(self):
        r = self.report
        self.assertEqual(r["fine_start"], r["fine_start_recount"])
        first = {"1": rules.URBAN_TIERS[0].min_excess, "0": rules.RURAL_TIERS[0].min_excess}
        self.assertEqual(r["fine_start"], {urban: [start - 1, start] for urban, start in first.items()})

    def test_limiter_and_distinct_input_counts_equal_an_independent_recount(self):
        r = self.report
        self.assertEqual(self.metrics["limiter_rows"], r["limiter_recount"])
        self.assertEqual(r["limiter_rows"], r["limiter_recount"])
        self.assertGreater(r["limiter_recount"], 0)
        self.assertEqual(r["distinct_inputs"], r["distinct_recount"])

    def test_every_remaining_mistake_sits_next_to_a_tier_boundary(self):
        self.assertTrue(self.report["mistake_distance"])
        self.assertLessEqual(max(self.report["mistake_distance"]), 2)

    def test_more_data_brings_the_model_to_the_law(self):
        curve, worst = self.report["curve"], self.report["curve_worst"]
        self.assertEqual(len(curve), 5)
        self.assertEqual(curve, sorted(curve))
        self.assertGreaterEqual(curve[-1], 0.999)
        self.assertTrue(all(w <= m for w, m in zip(worst, curve)))
        self.assertGreaterEqual(worst[-1], 0.999)

    def test_app_warns_exactly_where_the_model_disagrees_with_the_law(self):
        r = self.report
        self.assertEqual(r["mid_tier_warnings"], 0)
        for sentence, law_fine in zip(r["mistake_sentences"], r["mistake_laws"]):
            with self.subTest(sentence=sentence):
                self.assertIn(f"Uyarı: kanun tablosuna göre ceza {law_fine:,} TL".replace(",", "."), sentence)
                self.assertTrue(sentence.endswith("model bu kademe sınırının yakınında yanılıyor."))
        examples = r["example_sentences"]
        self.assertEqual(len(examples), 6)
        self.assertEqual(["Uyarı" in s for s in examples], [False] * 5 + [True])
        self.assertIn("kanun tablosuna göre ceza 20.000 TL ve 30 gün ehliyet geri alma", examples[-1])

    def test_weather_extremes_sit_in_small_groups(self):
        weather = self.report["weather"]
        self.assertEqual(set(weather), set(dataset.WEATHER_TYPES) | {"bilinmiyor"})
        largest = max(n for _, n in weather.values())
        highest = max(weather, key=lambda k: weather[k][0])
        lowest = min(weather, key=lambda k: weather[k][0])
        for group in (highest, lowest):
            self.assertLess(weather[group][1], largest / 4, group)

    def test_predictions_away_from_boundaries_equal_the_law(self):
        checks = self.report["mid_tier_checks"]
        self.assertEqual(len(checks), 33)
        self.assertTrue(all(fine_ok and days_ok for fine_ok, days_ok in checks))

    def test_sentences_for_the_driver(self):
        r = self.report
        self.assertEqual(r["no_fine"], "Otomobil, yerleşim yeri içi (sınır 50 km/s), 54 km/s: Ceza almazsınız.")
        self.assertIn("Ceza alırsınız. Tahminî ceza 6.000 TL", r["fine"])
        self.assertIn("1 ay içinde öderseniz 4.500 TL", r["fine"])
        self.assertNotIn("geri alınır", r["fine"])
        self.assertIn("20.000 TL", r["suspension"])
        self.assertTrue(r["suspension"].endswith("Ehliyetiniz 30 gün geri alınır."))

    def test_assistant_asks_again_after_wrong_answers(self):
        out = self.report["assistant_retry"]
        self.assertEqual(out.count("Bu seçeneklerden birini yazın."), 2)
        self.assertEqual(out.count("1 ile 300 arasında bir tam sayı yazın."), 2)
        self.assertIn("Otomobil, yerleşim yeri içi (sınır 50 km/s), 100 km/s: Ceza alırsınız. Tahminî ceza 20.000 TL", out)
        self.assertIn("Ehliyetiniz 30 gün geri alınır.", out)

    def test_assistant_accepts_codes_and_the_turkish_names_it_shows(self):
        r = self.report
        self.assertIn("Kamyon, otoyol (sınır 90 km/s), 105 km/s: Ceza alırsınız. Tahminî ceza 2.000 TL", r["assistant_names"])
        self.assertIn("Minibüs, yerleşim yeri içi (sınır 50 km/s), 77 km/s: Ceza alırsınız. Tahminî ceza 12.000 TL", r["assistant_labels"])
        self.assertNotIn("Bu seçeneklerden birini yazın.", r["assistant_labels"])
        upper = r["assistant_upper"]
        self.assertIn("Minibüs, otoyol (sınır 100 km/s), 126 km/s: Ceza alırsınız. Tahminî ceza 8.000 TL", upper)
        self.assertEqual(upper.count("1 ile 300 arasında bir tam sayı yazın."), 1)

    def test_assistant_says_when_it_falls_back_to_the_default(self):
        out = self.report["assistant_give_up"]
        self.assertEqual(out.count("Bu seçeneklerden birini yazın."), 3)
        self.assertIn("Üç denemede anlaşılamadı; Otomobil seçildi.", out)
        self.assertIn("Üç denemede anlaşılamadı; 104 km/s kullanıldı.", out)
        self.assertIn("Otomobil, yerleşim yeri içi (sınır 50 km/s), 104 km/s: Ceza alırsınız.", out)

    def test_sample_session_shows_each_question_with_its_answer_and_never_touches_the_keyboard(self):
        lines = self.report["assistant_typed"].splitlines()
        self.assertEqual(len(lines), 4)
        self.assertTrue(lines[0].startswith("Araç tipi [1 Otomobil") and lines[0].endswith("): Otomobil"))
        self.assertTrue(lines[1].startswith("Yol tipi [1 Yerleşim yeri içi") and lines[1].endswith("): bölünmüş yol"))
        self.assertTrue(lines[2].startswith("Radarın ölçtüğü hız") and lines[2].endswith("): 165"))
        self.assertTrue(lines[3].startswith("Otomobil, bölünmüş yol (sınır 110 km/s), 165 km/s: Ceza alırsınız."))

    def test_assistant_uses_default_answers_without_a_keyboard(self):
        out = self.report["assistant_default"]
        self.assertEqual(out.count("(otomatik cevap)"), 3)
        self.assertTrue(out.strip().endswith(self.report["default_sentence"]))

    def test_figures_and_metrics_are_written_and_match_the_committed_ones(self):
        expected = {"asim_ceza.png", "ceza_orani.png", "karisiklik_matrisi.png", "metrics.json", "model_karsilastirma.png", "ogrenme_egrisi.png"}
        self.assertEqual(set(self.report["figures"]), expected)
        committed = json.loads((ROOT / "figures" / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(self.metrics, committed)


class ColabBootstrap(unittest.TestCase):
    def test_empty_folder_fetches_the_project_and_runs_end_to_end(self):
        with tempfile.TemporaryDirectory() as folder:
            workdir = pathlib.Path(folder) / "content"
            workdir.mkdir()
            source = pathlib.Path(folder) / "source"
            copy_project(source)
            notebook = nbformat.read(build_notebook.NOTEBOOK, as_version=4)
            setup = nbformat.v4.new_code_cell(f"SOURCE = {str(source)!r}\n" + FAKE_CLONE)
            notebook.cells.insert(0, setup)
            notebook.cells.append(nbformat.v4.new_code_cell('print("PROBE_JSON:" + json.dumps({"calls": CLONE_CALLS, "cwd": os.getcwd()}))'))
            report = probe_report(execute(notebook, workdir))
            self.assertEqual(report["calls"], [["git", "clone", "--depth", "1", "https://github.com/AxelXoket/speeding-fine-predictor.git"]])
            self.assertEqual(pathlib.Path(report["cwd"]).resolve(), (workdir / "speeding-fine-predictor").resolve())
            produced = json.loads((workdir / "speeding-fine-predictor" / "figures" / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(produced, json.loads((ROOT / "figures" / "metrics.json").read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
