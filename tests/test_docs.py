"""The README states measured numbers and the fine table; both must match what the code measures and enforces."""
import importlib.util
import json
import pathlib
import re
import tempfile
import threading
import unittest
import zipfile

from pptx import Presentation
from pptx.util import Inches
from pypdf import PdfReader

from speeding_fine import dataset, rules

ROOT = pathlib.Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
METRICS = json.loads((ROOT / "figures" / "metrics.json").read_text(encoding="utf-8"))

_spec = importlib.util.spec_from_file_location("build_deliverables", ROOT / "tools" / "build_deliverables.py")
deliverables = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(deliverables)


class ReadmeNumbersAreMeasured(unittest.TestCase):
    def results_row(self, model_label):
        row = next(line for line in README.splitlines() if line.startswith(f"| {model_label} |"))
        return [cell.strip() for cell in row.strip("|").split("|")]

    def test_results_table_matches_metrics(self):
        for label, key in (("Model A", "Model A (ham)"), ("Model B", "Model B (aşım)")):
            with self.subTest(model=label):
                cells = self.results_row(label)
                scores = METRICS["models"][key]
                self.assertEqual(cells[2], deliverables.pct(scores["test_accuracy"]))
                self.assertEqual(cells[3], deliverables.pct(scores["law_agreement"]))

    def test_counts_in_the_text_match_metrics_and_generator(self):
        self.assertIn(f"({deliverables.thousands(METRICS['grid_rows'])} durum)", README)
        self.assertIn(f"{METRICS['rows_raw']} satırdır; temizlikten sonra {METRICS['rows_clean']} satır kalır", README)
        removed = (
            f"silinenler: {dataset.MISSING_SPEED_ROWS} boş hız, {dataset.IMPOSSIBLE_SPEED_ROWS} imkânsız hız, "
            f"{dataset.DUPLICATE_ROWS} tekrar eden satır."
        )
        fixed = f"Düzeltilenler: {dataset.MISSING_WEATHER_ROWS} boş hava durumu"
        self.assertIn(removed, README)
        self.assertIn(fixed, README)
        removed_rows = dataset.MISSING_SPEED_ROWS + dataset.IMPOSSIBLE_SPEED_ROWS + dataset.DUPLICATE_ROWS
        self.assertEqual(METRICS["rows_raw"] - removed_rows, METRICS["rows_clean"])
        grid = deliverables.thousands(METRICS["grid_rows"])
        self.assertIn(f"Model B, {grid} durumun {METRICS['model_b_disagreements']} tanesinde kanundan ayrılıyor; "
                      "bu durumların hepsi eğitimde görülmemiş girdilerde", README)
        self.assertEqual(METRICS["seen_agreement"], 1.0)
        self.assertIn(f"Eğitimde gördüğü girdilerde uyumu {deliverables.pct(METRICS['seen_agreement'])}, "
                      f"eğitim satırlarında hiç geçmeyen (aşım, yerleşim yeri içi mi) çiftlerinde "
                      f"{deliverables.pct(METRICS['unseen_agreement'])} "
                      f"(bu girdilerin payı {deliverables.pct(METRICS['unseen_share'])}).", README)
        self.assertIn(f"kanundaki {METRICS['boundaries_total']} kademe sınırından {METRICS['boundaries_found']} tanesini", README)
        self.assertEqual(METRICS["spelling_fixed"], dataset.MESSY_TEXT_ROWS)
        self.assertIn(f"{METRICS['spelling_fixed']} yazımı bozuk metin", README)
        speeds = METRICS["grid_rows"] // (len(rules.VEHICLE_TYPES) * len(rules.ROAD_TYPES))
        self.assertEqual(speeds, 25 + 95 + 1)
        self.assertIn(f"modelin {len(rules.VEHICLE_TYPES)} araç tipi, {len(rules.ROAD_TYPES)} yol tipi ve sınırın 25 km/s altından 95 km/s üstüne", README)
        self.assertIn(f"{METRICS['curve_seeds']} farklı veri tablosunun ortalamasında", README)
        self.assertIn(f"bu boyutta görülmemiş girdi payı {deliverables.pct(METRICS['curve_unseen_share'][-1])}", README)
        sizes = [size for size, *_ in METRICS["learning_curve"]]
        unseen = dict(zip(sizes, METRICS["curve_unseen_agreement"]))
        series = ", ".join(f"{deliverables.thousands(size)} kayıtta {deliverables.pct(value)}" for size, value in unseen.items())
        self.assertIn(f"Görülmemiş girdilerdeki uyum {series}.", README)
        self.assertIn(f"sınırlayıcı çalışırken görülemeyecek hızda {METRICS['limiter_rows']} kayıt var", README)
        biggest, mean, worst, _ = METRICS["learning_curve"][-1]
        self.assertIn(f"{deliverables.thousands(biggest)} kayda çıkınca uyum {deliverables.pct(mean)} oluyor", README)

    def test_mistake_distance_claim(self):
        self.assertIn(f"hepsi bir kademe sınırına en çok {METRICS['mistake_distance_max']} km/s uzaklıkta", README)
        self.assertLessEqual(METRICS["mistake_distance_max"], 2)
        boundaries = {1: [t.min_excess for t in rules.URBAN_TIERS], 0: [t.min_excess for t in rules.RURAL_TIERS]}
        for urban, excess in METRICS["model_b_mistakes"]:
            nearest = min(min(abs(excess - b), abs(excess - (b - 1))) for b in boundaries[urban])
            self.assertLessEqual(nearest, 2, (urban, excess))


class ReadmeFineTableMatchesTheRules(unittest.TestCase):
    def test_every_row_matches_both_tier_tables(self):
        pattern = re.compile(r"^\| (\d+)(?:-(\d+)| ve üstü) \| (\d+)(?:-(\d+)| ve üstü) \| ([\d.]+) TL \|(?: (\d+) gün)? \|$")
        rows = [pattern.match(line) for line in README.splitlines()]
        rows = [r for r in rows if r]
        self.assertEqual(len(rows), len(rules.URBAN_TIERS))
        for match, urban, rural in zip(rows, rules.URBAN_TIERS, rules.RURAL_TIERS):
            u_min, u_max, r_min, r_max, fine, days = match.groups()
            self.assertEqual((int(u_min), int(u_max) if u_max else None), (urban.min_excess, urban.max_excess))
            self.assertEqual((int(r_min), int(r_max) if r_max else None), (rural.min_excess, rural.max_excess))
            self.assertEqual(int(fine.replace(".", "")), urban.fine_tl)
            self.assertEqual(urban.fine_tl, rural.fine_tl)
            self.assertEqual(int(days) if days else 0, urban.suspension_days)


TEMPLATE_LEFTOVERS = ("Steve Canny", "Macintosh", "4:3", "screen4x3", "generated using python-pptx", "2013-",
                      "thumbnail.jpeg", "printerSettings1.bin")


def template_leftovers(path: pathlib.Path) -> list[str]:
    """python-pptx template metadata or parts still present anywhere in the package."""
    with zipfile.ZipFile(path) as package:
        names = package.namelist()
        text = " ".join(names + [package.read(n).decode("utf-8", "replace") for n in names if n.endswith((".xml", ".rels"))])
    return [word for word in TEMPLATE_LEFTOVERS if word in text]


class SlidesNeverWrapOnTheirOwn(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        missing = [str(p) for p in deliverables.FONT_FILES.values() if not p.exists()]
        if missing:
            raise unittest.SkipTest(f"the slide width gate measures with Calibri, not found at {missing} (Windows only)")
        cls.folder = tempfile.TemporaryDirectory()
        try:
            cls.path = deliverables.build_slides(deliverables.load_metrics(), pathlib.Path(cls.folder.name) / "deck.pptx")
        except SystemExit as refusal:
            # The builder's own gate stops a bad deck; report it as a test error instead of ending the whole run.
            raise AssertionError(f"slide build refused: {refusal}") from None
        cls.deck = Presentation(str(cls.path))

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_real_deck_has_no_line_wider_than_its_box(self):
        deck = self.deck
        self.assertEqual(len(deck.slides), 5)
        self.assertEqual(deliverables.overflowing_lines(deck), [])
        texts = [run.text for slide in deck.slides for shape in slide.shapes if shape.has_text_frame
                 for paragraph in shape.text_frame.paragraphs for run in paragraph.runs]
        self.assertGreater(len(texts), 40)
        self.assertIn("MİNİ UYGULAMA VE ÇIKARIM", texts)
        self.assertFalse([t for t in texts if re.search(r"\d \S", t)], "a number is separated from its unit")

    def test_deck_metadata_names_only_the_handle(self):
        props = self.deck.core_properties
        self.assertEqual((props.author, props.last_modified_by), ("AxelXoket", "AxelXoket"))
        self.assertEqual(props.title, "Hız Cezası Tahmini")
        self.assertEqual(props.comments, "")
        self.assertGreaterEqual(props.created.year, 2026)
        self.assertEqual(template_leftovers(self.path), [])
        self.assertEqual((self.deck.slide_width, self.deck.slide_height), (Inches(13.333), Inches(7.5)))

    def test_leftover_scan_finds_the_template_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            plain = pathlib.Path(folder) / "plain.pptx"
            Presentation().save(str(plain))
            self.assertTrue({"Steve Canny", "Macintosh", "screen4x3", "thumbnail.jpeg", "printerSettings1.bin"} <= set(template_leftovers(plain)))

    def test_slide_examples_come_from_the_saved_notebook_session(self):
        session = deliverables.sample_session()
        self.assertEqual([line.rsplit("): ", 1)[1] for line in session[:3]], ["Otomobil", "bölünmüş yol", "165"])
        self.assertTrue(session[0].startswith("Araç tipi [...] (varsayılan: Otomobil)"))
        self.assertEqual(session[3], METRICS["examples"][3])
        texts = [shape.text_frame.text for shape in self.deck.slides[4].shapes if shape.has_text_frame]
        self.assertTrue(any(session[1] in text for text in texts))

    def test_committed_deck_has_the_same_text_as_a_fresh_build(self):
        def slide_texts(deck):
            return [[shape.text_frame.text for shape in slide.shapes if shape.has_text_frame] for slide in deck.slides]

        self.assertEqual(slide_texts(Presentation(str(deliverables.DOCS / "sunum.pptx"))), slide_texts(self.deck))

    def test_gate_reports_a_line_that_is_too_long(self):
        deck = Presentation()
        slide = deck.slides.add_slide(deck.slide_layouts[6])
        deliverables.text_box(slide, 0, 0, Inches(3), Inches(1), [("Bu satır üç inçlik bir kutuya asla sığmayacak kadar uzun", 18, False, deliverables.INK)])
        deliverables.text_box(slide, 0, Inches(2), Inches(3), Inches(1), [("Kısa satır\nİkinci kısa satır", 18, False, deliverables.INK)])
        problems = deliverables.overflowing_lines(deck)
        self.assertEqual(len(problems), 1)
        self.assertIn("asla sığmayacak", problems[0])

    def test_turkish_upper_case_keeps_the_dot(self):
        self.assertEqual(deliverables.tr_upper("Keşifçi veri analizi"), "KEŞİFÇİ VERİ ANALİZİ")
        self.assertEqual(deliverables.tr_upper("ılık"), "ILIK")

    def test_output_lines_put_the_warning_on_its_own_line(self):
        prompt, answer = deliverables.output_lines(METRICS["examples"][5])
        self.assertEqual(prompt, "Otomobil, yerleşim yeri içi (sınır 50 km/s), 96 km/s:")
        first, warning = answer.split("\n")
        self.assertTrue(first.startswith("Ceza alırsınız."))
        self.assertTrue(warning.startswith("Uyarı: kanun tablosuna göre ceza 20.000 TL"))


class CommittedReportMatchesMetrics(unittest.TestCase):
    def test_report_pdf_states_the_measured_numbers(self):
        text = " ".join(page.extract_text() for page in PdfReader(ROOT / "docs" / "rapor.pdf").pages)
        text = " ".join(text.replace(" ", " ").split())
        engineered = METRICS["models"]["Model B (aşım)"]
        expected = [
            deliverables.pct(engineered["law_agreement"]),
            f"{deliverables.thousands(METRICS['grid_rows'])} durumun {METRICS['model_b_disagreements']} tanesinde",
            f"{METRICS['boundaries_found']} tanesini tam yerinde",
            f"bu girdilerde uyum {deliverables.pct(METRICS['unseen_agreement'])}",
            f"görülmemiş girdi payı {deliverables.pct(METRICS['curve_unseen_share'][-1])}",
            METRICS["examples"][3].split(": ", 1)[1],
        ]
        self.assertEqual([phrase for phrase in expected if phrase not in text], [])


class LineBreakGlue(unittest.TestCase):
    def test_numbers_stay_with_their_unit_and_article_numbers_with_their_label(self):
        glue = lambda text: deliverables.keep_numbers_with_units(text, "_")  # noqa: E731
        self.assertEqual(glue("ceza 2.000 TL ile 30.000 TL arasında"), "ceza 2.000_TL_ile_30.000_TL arasında")
        self.assertEqual(glue("KTK md. 51/2; Yönetmelik md. 100"), "KTK md._51/2; Yönetmelik md._100")
        self.assertEqual(glue("KTK 51/3 ve 51/4"), "KTK_51/3_ve 51/4")
        self.assertEqual(glue("Model B ehliyet geri alınır"), "Model_B ehliyet geri_alınır")
        self.assertEqual(glue("30 ile 90 gün"), "30_ile_90_gün")
        self.assertEqual(glue("eğitimde %100, testte %86,5"), "eğitimde_%100, testte_%86,5")
        self.assertEqual(glue("9'ar kademe sınırı; bu 18 sınırın 13 tanesini"), "9'ar kademe_sınırı; bu 18_sınırın 13_tanesini")
        self.assertEqual(glue("15 boş hava durumu; sınırın iki yanında"), "15_boş_hava_durumu; sınırın iki_yanında")
        self.assertEqual(glue("ne kadar doğru; teste 148 satır (%25)"), "ne_kadar doğru; teste 148_satır_(%25)")
        self.assertEqual(glue("%75 eğitim (443 satır), sınavın cevap anahtarı"), "%75_eğitim_(443_satır), sınavın cevap_anahtarı")

    def test_plain_words_are_left_alone(self):
        text = "veri kanundan üretildi; ölçüm hatası yok"
        self.assertEqual(deliverables.keep_numbers_with_units(text, "_"), text)


class ReportPrintWaitsForEdge(unittest.TestCase):
    """Edge's launcher exits before the PDF is written, so the build must wait for the new file and never read the old one."""

    WHOLE = b"%PDF-1.4\nnew\n%%EOF\n"

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.pdf = pathlib.Path(folder.name) / "rapor.pdf"
        self.html = pathlib.Path(folder.name) / "rapor.html"

    def late_writer(self, content: bytes):
        """A stand-in for Edge: returns at once and writes the PDF a moment later, as the real launcher does."""
        def launch(*args, **kwargs):
            writer = threading.Timer(0.15, self.pdf.write_bytes, (content,))
            self.addCleanup(writer.cancel)
            writer.start()
        return launch

    def test_the_new_pdf_is_read_not_the_previous_build(self):
        self.pdf.write_bytes(b"%PDF-1.4\nold\n%%EOF\n")
        deliverables.print_pdf(self.html, self.pdf, launch=self.late_writer(self.WHOLE), timeout=2, poll=0.02)
        self.assertEqual(self.pdf.read_bytes(), self.WHOLE)

    def test_a_pdf_that_never_appears_stops_the_build(self):
        with self.assertRaises(SystemExit):
            deliverables.print_pdf(self.html, self.pdf, launch=lambda *a, **k: None, timeout=0.1, poll=0.02)

    def test_a_half_written_pdf_is_not_accepted(self):
        with self.assertRaises(SystemExit):
            deliverables.print_pdf(self.html, self.pdf, launch=self.late_writer(b"%PDF-1.4\n1 0 obj"), timeout=0.4, poll=0.02)

    def test_an_unreadable_pdf_means_wait_not_crash(self):
        self.pdf.mkdir()  # reading a folder raises OSError, like a file Windows still holds open
        with self.assertRaises(SystemExit):
            deliverables.wait_for_pdf(self.pdf, timeout=0.1, poll=0.02)


class PercentFormat(unittest.TestCase):
    def test_turkish_percent(self):
        self.assertEqual(deliverables.pct(0.8649), "%86,5")
        self.assertEqual(deliverables.pct(1.0), "%100")
        self.assertEqual(deliverables.pct(0.9995), "%100")
        self.assertEqual(deliverables.pct(0.05), "%5")
        self.assertEqual(deliverables.thousands(3388), "3.388")


if __name__ == "__main__":
    unittest.main()
