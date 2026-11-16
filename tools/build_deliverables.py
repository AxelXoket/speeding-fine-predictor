"""Build the one-page report (docs/rapor.pdf) and the slides (docs/sunum.pptx) from figures/metrics.json.

Usage: python tools/build_deliverables.py
Every number comes from figures/metrics.json, which the notebook writes; run tools/build_notebook.py first.
The PDF is printed by headless Edge (no window, throwaway profile).
"""
import datetime
import json
import pathlib
import re
import subprocess
import sys
import tempfile
import time
import zipfile

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
from PIL import ImageFont
from pypdf import PdfReader

ROOT_FOR_IMPORT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_FOR_IMPORT))
from speeding_fine import dataset, rules  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"
DOCS = ROOT / "docs"
EDGE = pathlib.Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
REPO_URL = "https://github.com/AxelXoket/speeding-fine-predictor"
COLAB_URL = "https://colab.research.google.com/github/AxelXoket/speeding-fine-predictor/blob/main/speeding_fine_prediction.ipynb"
HANDLE = "AxelXoket"
AUTHOR_LINE = f"{HANDLE} · Kasım 2026"
DECK_TITLE = "Hız Cezası Tahmini"
NOTEBOOK = ROOT / "speeding_fine_prediction.ipynb"
OPTION_LIST = re.compile(r" \[[^\]]*\]")
APP_PROPERTIES = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">'
    "<Application>python-pptx</Application><PresentationFormat>Geniş ekran (16:9)</PresentationFormat>"
    "<Slides>{slides}</Slides></Properties>"
)
TEMPLATE_PARTS = ("docProps/thumbnail.jpeg", "ppt/printerSettings/printerSettings1.bin")
TEMPLATE_RELATIONSHIP = re.compile(r'<Relationship [^>]*Target="[^"]*(?:thumbnail\.jpeg|printerSettings1\.bin)"[^>]*/>')

NAVY = RGBColor(0x1F, 0x3A, 0x5F)
RED = RGBColor(0xC0, 0x39, 0x2B)
GREEN = RGBColor(0x1E, 0x8E, 0x4E)
INK = RGBColor(0x22, 0x2B, 0x36)
MUTED = RGBColor(0x5D, 0x6D, 0x7E)
PANEL = RGBColor(0xF2, 0xF5, 0xF8)
FONT = "Calibri"


def pct(value: float) -> str:
    """Turkish percent with one decimal, without a trailing ',0' (so '%100' reads naturally)."""
    text = f"{value * 100:.1f}"
    return "%" + (text[:-2] if text.endswith(".0") else text).replace(".", ",")


def thousands(value: int) -> str:
    return f"{value:,}".replace(",", ".")


NO_BREAK = "\u00a0"
DIGIT_SPACE = re.compile(r"(?<=\d) (?=[^\s<])")
FIXED_PHRASES = ("geri alınır", "geri alınıyor", "Model A", "Model B", "km/s ile", "ölçülen hız", "aşılan km/s", "tam yerinde",
                 "kademe sınır", "boş hava durumu", "hava durumu", "Bu durumların", "bu girdilerde", "farklı veri",
                 "cevap anahtarı", "iki yanında", "ne kadar")
# A label before its number stays with it: 'KTK 51/2', 'md. 100', 'testte %86,5', 'satır (%75)'.
ABBREVIATION_NUMBER = re.compile(
    r"(?<=\b[A-ZÇĞİÖŞÜ]{2}) (?=\d)|(?<=\b[A-ZÇĞİÖŞÜ]{3}) (?=\d)|(?<=\bmd\.) (?=\d)|(?<=\w) (?=%\d)|(?<=\w) (?=\(%?\d)"
)
NUMBER_RANGE = re.compile(r"(?:(?<=\d)|(?<=TL)) ile (?=\d)")


def keep_numbers_with_units(text: str, space: str = NO_BREAK) -> str:
    """Glue a number to the word after it ('6 boş hız', '20.000 TL') and keep fixed phrases on one line."""
    for phrase in FIXED_PHRASES:
        text = text.replace(phrase, phrase.replace(" ", space))
    text = NUMBER_RANGE.sub(f"{space}ile{space}", text)
    text = ABBREVIATION_NUMBER.sub(space, text)
    return DIGIT_SPACE.sub(space, text)


def load_metrics() -> dict:
    metrics = json.loads((FIGURES / "metrics.json").read_text(encoding="utf-8"))
    raw, engineered = metrics["models"]["Model A (ham)"], metrics["models"]["Model B (aşım)"]
    curve = {size: mean for size, mean, _, _ in metrics["learning_curve"]}
    return {
        **metrics,
        "raw": raw,
        "eng": engineered,
        "curve": curve,
        "max_curve_size": max(curve),
        "mistake_count": len(metrics["model_b_mistakes"]),
    }


# ---------- report ----------

REPORT_HTML = """<!doctype html>
<html lang="tr"><head><meta charset="utf-8"><title>Hız Cezası Tahmini - Proje Raporu</title>
<style>
@page {{ size: A4; margin: 10mm 14mm; }}
* {{ box-sizing: border-box; }}
body {{ font-family: "Segoe UI", Calibri, Arial, sans-serif; color: #222b36; font-size: 9.4pt; line-height: 1.34; margin: 0; }}
h1 {{ font-size: 18pt; color: #1f3a5f; margin: 0 0 1mm; }}
.sub {{ color: #5d6d7e; font-size: 9pt; margin-bottom: 3mm; }}
.sub a {{ color: #1f3a5f; }}
h2 {{ font-size: 10.5pt; color: #c0392b; margin: 3mm 0 1mm; text-transform: uppercase; letter-spacing: .04em; }}
p {{ margin: 0 0 1.5mm; }}
p, li, .cap, .take, .sub {{ text-wrap: pretty; hyphens: manual; }}
h1 {{ text-wrap: balance; }}
code {{ white-space: nowrap; }}
ul {{ margin: 0 0 1.5mm; padding-left: 4.5mm; }}
li {{ margin-bottom: .6mm; }}
.cols {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6mm; }}
.figs {{ display: grid; grid-template-columns: 1fr 1fr; gap: 4mm; margin-top: 2mm; align-items: center; }}
.figs img {{ width: 72%; display: block; margin: 0 auto; }}
.cap {{ font-size: 8pt; color: #5d6d7e; margin-top: .5mm; }}
table {{ border-collapse: collapse; width: 100%; font-size: 9pt; margin: 1mm 0 1.5mm; }}
th, td {{ border-bottom: 1px solid #d5dde5; padding: 1.2mm 1.5mm; text-align: left; }}
th {{ background: #f2f5f8; color: #1f3a5f; }}
td.n, th.n {{ text-align: right; font-variant-numeric: tabular-nums; }}
td:first-child {{ white-space: nowrap; }}
.quote {{ font-size: 8.8pt; background: #1b2633; color: #e6ecf2; padding: 2mm 3mm; border-radius: 1.5mm; white-space: pre-line; }}
.quote b {{ color: #8fd6a8; }}
.take {{ background: #f2f5f8; border-left: 3px solid #1e8e4e; padding: 2mm 3mm; margin-top: 2mm; }}
.foot {{ margin-top: 2.5mm; font-size: 8pt; color: #5d6d7e; }}
code {{ font-family: Consolas, monospace; font-size: 8.6pt; }}
</style></head><body>
<h1>Hız Cezası Tahmini: Ceza Yer miyim, Yersem Ne Kadar?</h1>
<div class="sub">Uçtan Uca Yapay Zeka Projesi: Veriden Tahmine · Yapay Zeka İçin Basitleştirilmiş Python ·
<a href="{repo}">GitHub</a> · <a href="{colab}">Colab not defteri</a></div>

<div class="cols">
<div>
<h2>Problem</h2>
<p>Araç tipi, yol tipi ve radarın ölçtüğü hızdan sürücünün ceza alıp almayacağını, alırsa kaç TL ödeyeceğini ve ehliyetinin geri alınıp alınmayacağını tahmin etmek. Hedef&nbsp;<code>ceza_tutari_tl</code>; her tutar bir sınıf olduğu için problem bir <b>sınıflandırma</b>.</p>
<p>2026'da hız cezaları sınırın kaç km/s aşıldığına göre 9 kademeye bağlandı: ceza 2.000 TL ile 30.000 TL arasında, yüksek aşımlarda ehliyet 30 ile 90 gün geri alınıyor. Tahmin, yola çıkmadan riskini görmek isteyen sürücüye, kuralları anlatan sürücü kursuna ve şoförlerini eğiten filo yöneticisine yarar.</p>
</div>
<div>
<h2>Veri</h2>
<ul>
<li>Kişi bazında gerçek ceza kaydı kamuya açık değil; veriyi kanundan ürettim. Kanun bu yüzden sınavın cevap anahtarı: soru, modelin kuralı örneklerden ne kadar doğru öğrendiği.</li>
<li>{rows_raw} satır, 9 sütun. Bilerek bozulan satırlar temizlendi: {missing_speed} boş hız, {impossible} imkânsız hız ve {duplicates} tekrar silindi; {missing_weather} boş hava durumu "bilinmiyor" yapıldı, {messy} bozuk yazım düzeltildi. Kalan: <b>{rows_clean} satır</b>.</li>
<li>Modeller, kanunun kendisiyle {grid_rows} durumda ({vehicles} araç tipi, {roads} yol tipi, sınırın 25 km/s altından 95 km/s üstüne her hız) karşılaştırılıyor. <b>Kanunla uyum</b>, aynı cezanın verildiği durumların oranı.</li>
</ul>
</div>
</div>

<div class="figs">
<div><img src="../figures/asim_ceza.png" alt="Aşım ve ceza grafiği"><div class="cap">Yerleşim yeri içinde ceza basamakları daha erken başlıyor.</div></div>
<div><img src="../figures/model_karsilastirma.png" alt="Model karşılaştırması"><div class="cap">Aynı yöntem, doğru özelliklerle kanuna çok daha yakın.</div></div>
</div>

<div class="cols">
<div>
<h2>Model</h2>
<p>Kanun bir eşik merdiveni, karar ağacı da eşik sorularıyla çalışıyor; öğrendiği eşikler okunabiliyor. Eğitime {train_rows} satır (%75), teste {test_rows} satır (%25) ayrıldı.</p>
<ul>
<li><b>Model A (ham):</b> araç tipi, yol tipi, hava, hız sınırı, ölçülen hız, saat.</li>
<li><b>Model B (aşım):</b> keşifte belirleyici çıkan iki sütun, aşılan km/s ve yerleşim yeri içi mi.</li>
</ul>
<p>Kanunda yerleşim yeri içi ve dışı için 9'ar kademe sınırı var; Model B bu {boundaries_total} sınırın {boundaries_found} tanesini tam yerinde, yani sınırın iki yanında kanunla aynı cezayla buldu. Eğitimde hiç görülmemiş (aşım, yerleşim yeri içi mi) çiftlerinin payı {unseen_share}; bu girdilerde uyum {unseen}, görülenlerde {seen}.</p>
</div>
<div>
<h2>Sonuç</h2>
<table>
<tr><th></th><th class="n">Eğitim<br>doğruluğu</th><th class="n">Test<br>doğruluğu</th><th class="n">Kanunla<br>uyum</th></tr>
<tr><td>Model A (ham)</td><td class="n">{raw_train}</td><td class="n">{raw_test}</td><td class="n">{raw_law}</td></tr>
<tr><td>Model B (aşım)</td><td class="n">{eng_train}</td><td class="n">{eng_test}</td><td class="n">{eng_law}</td></tr>
</table>
<p>Model A eğitim verisini ezberliyor: eğitimde {raw_train}, testte {raw_test}; bu fark aşırı öğrenmenin ölçüsü. Model B, {grid_rows} durumun {disagreements} tanesinde kanundan ayrılıyor. Bu durumların hepsi eğitimde görülmemiş {mistake_count} aşım değerinde ve her biri bir kademe sınırına en çok {distance} km/s uzaklıkta. {seeds} farklı veri tablosunun ortalamasında, eğitim verisi {max_size} kayda çıkınca uyum {max_curve} oluyor; bu boyutta görülmemiş girdi payı {max_unseen_share}.</p>
</div>
</div>

<h2>Mini uygulama</h2>
<p>Aşama 6 araç tipini, yol tipini ve hızı sorar. Örnek çalıştırmada cevaplar {session_answers}; çıktı:</p>
<p class="quote"><b>{example_prompt}</b>
{example_answer}</p>

<div class="take"><b>Çıkarım:</b> Aynı algoritma, aldığı bilgiye göre ya eğitim verisini ezberliyor ya da kanunun kuralına yaklaşıyor.<br><b>Zayıf yönler:</b> veri kanundan üretildi; ölçüm hatası ve yerel levha sınırları yok.<br>Tutarların 2027'den itibaren yeniden değerlemeyle artması bekleniyor; bunu doğrulayan resmî duyuru bulamadım.</div>
<div class="foot"><b>Dayanaklar:</b> ceza tutarları 2918 sayılı Karayolları Trafik Kanunu (KTK) md. 51/2, 7574 sayılı Kanunla değişik (Resmî Gazete 27.02.2026).<br>Ehliyet yaptırımı KTK md. 51/3 ve 51/4; hız sınırları Karayolları Trafik Yönetmeliği md. 100; 1 aylık ödeme süresi KTK md. 115/1.<br>Süresinde ödemede %25 indirim Kabahatler Kanunu md. 17/6; yeniden değerleme KTK Ek Madde 3 ve Kabahatler Kanunu md. 17/7.<br>{author}</div>
</body></html>
"""


def wait_for_pdf(path: pathlib.Path, timeout: float = 60.0, poll: float = 0.25) -> None:
    """Edge's launcher exits before the print is written (measured: 0.06 s vs 0.5 s); wait until the PDF trailer is in."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if path.read_bytes().rstrip().endswith(b"%%EOF"):
                return
        except OSError:  # not there yet, or Windows still holds it open for writing
            pass
        time.sleep(poll)
    raise SystemExit(f"Edge did not finish {path.name} within {timeout:g} s")


def print_pdf(html_path: pathlib.Path, pdf_path: pathlib.Path, launch=subprocess.run, timeout: float = 60.0,
              poll: float = 0.25) -> None:
    """Print a page with headless Edge (no window, throwaway profile) and return once the new PDF is whole."""
    # The previous build's PDF goes first, so nothing after this can read a stale file.
    pdf_path.unlink(missing_ok=True)
    # Edge's crash handler can hold the throwaway profile for a moment after exit; a leftover temp folder is harmless.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as profile:
        launch(
            [str(EDGE), "--headless=new", "--disable-gpu", "--no-first-run", f"--user-data-dir={profile}",
             "--no-pdf-header-footer", f"--print-to-pdf={pdf_path}", html_path.as_uri()],
            check=True, capture_output=True, timeout=120,
        )
        wait_for_pdf(pdf_path, timeout, poll)


def build_report(m: dict) -> pathlib.Path:
    if m["seen_agreement"] != 1.0:
        raise SystemExit("the report says every disagreement is on an unseen input, but the metrics disagree")
    session = sample_session()
    answers = [f'"{line.rsplit("): ", 1)[1]}"' for line in session[:3]]
    example_prompt, example_answer = output_lines(session[3])
    html = REPORT_HTML.format(
        session_answers=", ".join(answers[:-1]) + " ve " + answers[-1],
        disagreements=m["model_b_disagreements"],
        boundaries_total=m["boundaries_total"],
        boundaries_found=m["boundaries_found"],
        seen=pct(m["seen_agreement"]),
        unseen_share=pct(m["unseen_share"]),
        seeds=m["curve_seeds"],
        max_unseen_share=pct(m["curve_unseen_share"][-1]),
        distance=m["mistake_distance_max"],
        unseen=pct(m["unseen_agreement"]),
        repo=REPO_URL,
        colab=COLAB_URL,
        rows_raw=m["rows_raw"],
        rows_clean=m["rows_clean"],
        missing_speed=m["missing_speed"],
        missing_weather=m["missing_weather"],
        duplicates=m["duplicates"],
        grid_rows=thousands(m["grid_rows"]),
        train_rows=m["train_rows"],
        test_rows=m["test_rows"],
        raw_train=pct(m["raw"]["train_accuracy"]),
        raw_test=pct(m["raw"]["test_accuracy"]),
        raw_law=pct(m["raw"]["law_agreement"]),
        eng_train=pct(m["eng"]["train_accuracy"]),
        eng_test=pct(m["eng"]["test_accuracy"]),
        eng_law=pct(m["eng"]["law_agreement"]),
        mistake_count=m["mistake_count"],
        max_size=thousands(m["max_curve_size"]),
        max_curve=pct(m["curve"][m["max_curve_size"]]),
        example_prompt=example_prompt,
        example_answer=example_answer,
        impossible=dataset.IMPOSSIBLE_SPEED_ROWS,
        messy=m["spelling_fixed"],
        vehicles=len(rules.VEHICLE_TYPES),
        roads=len(rules.ROAD_TYPES),
        author=AUTHOR_LINE,
    )
    head, body = html.split("<body>", 1)
    html = head + "<body>" + keep_numbers_with_units(body, "&nbsp;")
    DOCS.mkdir(exist_ok=True)
    html_path = DOCS / "rapor.html"
    html_path.write_text(html, encoding="utf-8")
    pdf_path = DOCS / "rapor.pdf"
    print_pdf(html_path, pdf_path)
    pages = len(PdfReader(pdf_path).pages)
    if pages != 1:
        raise SystemExit(f"the report must fit on one page; Edge printed {pages}")
    return pdf_path


# ---------- slides ----------
# Every line on a slide is broken by hand at a natural phrase boundary and measured with the real Calibri
# metrics, so no renderer (PowerPoint, LibreOffice, Canva) ever wraps a line on its own.

FONT_FILES = {False: pathlib.Path(r"C:\Windows\Fonts\calibri.ttf"), True: pathlib.Path(r"C:\Windows\Fonts\calibrib.ttf")}
TEXT_INSET_IN = 0.2  # python-pptx text boxes keep 0.1 inch on each side
FIT_MARGIN = 0.95  # a line may use at most 95% of the usable width


def tr_upper(text: str) -> str:
    """Turkish upper case: 'veri' -> 'VERİ', 'ılık' -> 'ILIK'."""
    return text.replace("i", "İ").replace("ı", "I").upper()


def line_width_pt(text: str, size: float, bold: bool) -> float:
    font = ImageFont.truetype(str(FONT_FILES[bold]), 100)
    return font.getlength(text) * size / 100


def overflowing_lines(deck: Presentation) -> list[str]:
    """Every text line that would not fit on one line inside its box."""
    problems = []
    for number, slide in enumerate(deck.slides, start=1):
        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            usable_pt = (shape.width / 914400 - TEXT_INSET_IN) * 72 * FIT_MARGIN
            for paragraph in shape.text_frame.paragraphs:
                for run in paragraph.runs:
                    for line in run.text.split("\v"):
                        width = line_width_pt(line, run.font.size.pt, bool(run.font.bold))
                        if width > usable_pt:
                            problems.append(f"slide {number}: {width:.0f}pt > {usable_pt:.0f}pt: {line!r}")
    return problems


def text_box(slide, left, top, width, height, paragraphs, align=PP_ALIGN.LEFT):
    """paragraphs: (text, size, bold, color); '\\n' in text is a manual line break, empty text a small gap."""
    box = slide.shapes.add_textbox(left, top, width, height)
    frame = box.text_frame
    frame.word_wrap = True
    for index, (text, size, bold, color) in enumerate(paragraphs):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.alignment = align
        paragraph.space_after = Pt(size * 0.45)
        for line_index, line in enumerate(text.split("\n")):
            if line_index:
                paragraph.add_line_break()
            run = paragraph.add_run()
            run.text = keep_numbers_with_units(line)
            run.font.name, run.font.size, run.font.bold, run.font.color.rgb = FONT, Pt(size), bold, color
    return box


def panel(slide, left, top, width, height, fill=PANEL):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.adjustments[0] = 0.04
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.fill.background()
    shape.shadow.inherit = False
    return shape


def header(slide, kicker, title, number, total):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.18), Inches(7.5))
    bar.fill.solid()
    bar.fill.fore_color.rgb = RED
    bar.line.fill.background()
    text_box(slide, Inches(0.6), Inches(0.35), Inches(11), Inches(0.4), [(tr_upper(kicker), 12, True, RED)])
    text_box(slide, Inches(0.6), Inches(0.7), Inches(12), Inches(0.9), [(title, 30, True, NAVY)])
    text_box(slide, Inches(11.2), Inches(6.95), Inches(1.8), Inches(0.4), [(f"{number} / {total}", 11, False, MUTED)], PP_ALIGN.RIGHT)


def picture(slide, path, left, top, width, description):
    """Add a chart with alt text that says what it shows (python-pptx would use the file name)."""
    shape = slide.shapes.add_picture(str(path), left, top, width=width)
    shape._element.nvPicPr.cNvPr.set("descr", description)
    return shape


def output_lines(sentence: str) -> tuple[str, str]:
    """'Araç, yol (sınır N km/s), S km/s: Ceza ... Uyarı: ...' -> situation line, answer lines (warning on its own line)."""
    situation, answer = sentence.split(": ", 1)
    return situation + ":", answer.replace(" Uyarı:", "\nUyarı:")


def sample_session() -> list[str]:
    """The saved output of the notebook's sample session, with the long option lists shortened to '[...]'."""
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    cell = next(c for c in cells if c["cell_type"] == "code" and "".join(c["source"]).startswith("# Sample session"))
    text = "".join("".join(output.get("text", "")) for output in cell["outputs"])
    return [OPTION_LIST.sub(" [...]", line) for line in text.splitlines()]


def set_document_properties(deck: Presentation) -> None:
    """Replace python-pptx's template metadata (another author, a 2013 date, a 4:3 size type) with this deck's own."""
    props = deck.core_properties
    props.title = DECK_TITLE
    props.author = props.last_modified_by = HANDLE
    props.subject = props.keywords = props.comments = props.category = ""
    props.revision = 1
    props.created = props.modified = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None, microsecond=0)
    # The template's slide size says 4:3 even after the size changes; without a type attribute the size is custom.
    deck._element.sldSz.attrib.pop("type", None)


def clean_package(path: pathlib.Path, slide_count: int) -> None:
    """Drop what python-pptx copies from its template: a blank 4:3 thumbnail, Apple printer settings, its app.xml."""
    with zipfile.ZipFile(path) as source:
        entries = [(info, source.read(info.filename)) for info in source.infolist() if info.filename not in TEMPLATE_PARTS]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as target:
        for info, data in entries:
            if info.filename == "docProps/app.xml":
                data = APP_PROPERTIES.format(slides=slide_count).encode("utf-8")
            elif info.filename.endswith(".rels"):
                data = TEMPLATE_RELATIONSHIP.sub("", data.decode("utf-8")).encode("utf-8")
            target.writestr(info, data)


def build_slides(m: dict, path: pathlib.Path = DOCS / "sunum.pptx") -> pathlib.Path:
    deck = Presentation()
    deck.slide_width, deck.slide_height = Inches(13.333), Inches(7.5)
    blank = deck.slide_layouts[6]
    total = 5
    raw, eng = m["raw"], m["eng"]

    # 1. Title
    s = deck.slides.add_slide(blank)
    cover = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, deck.slide_width, deck.slide_height)
    cover.fill.solid()
    cover.fill.fore_color.rgb = NAVY
    cover.line.fill.background()
    accent = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(2.05), Inches(1.2), Inches(0.08))
    accent.fill.solid()
    accent.fill.fore_color.rgb = RED
    accent.line.fill.background()
    white = RGBColor(0xFF, 0xFF, 0xFF)
    soft = RGBColor(0xC9, 0xD6, 0xE3)
    text_box(s, Inches(0.8), Inches(2.3), Inches(11.5), Inches(1.2), [("Hız Cezası Tahmini", 48, True, white)])
    text_box(s, Inches(0.8), Inches(3.4), Inches(11.5), Inches(0.8), [("Ceza yer miyim, yersem ne kadar?", 28, False, soft)])
    text_box(s, Inches(0.8), Inches(4.5), Inches(11.5), Inches(1.2), [
        ("Uçtan Uca Yapay Zeka Projesi: Veriden Tahmine", 16, True, white),
        ("Yapay Zeka İçin Basitleştirilmiş Python · Bitirme projesi", 14, False, soft),
    ])
    text_box(s, Inches(0.8), Inches(6.4), Inches(11.5), Inches(0.6), [(f"{REPO_URL.replace('https://', '')}   ·   {AUTHOR_LINE}", 12, False, soft)])

    # 2. Problem and data
    s = deck.slides.add_slide(blank)
    header(s, "Problem ve veri", "Radar ölçümünden ceza ve tutar tahmini", 2, total)
    panel(s, Inches(0.6), Inches(1.75), Inches(5.9), Inches(4.95))
    text_box(s, Inches(0.9), Inches(1.95), Inches(5.4), Inches(4.6), [
        ("Problem", 18, True, NAVY),
        ("Araç tipi, yol tipi ve ölçülen hızdan:\nceza var mı, kaç TL, ehliyet geri alınır mı?", 15, False, INK),
        ("Hedef ve tür", 18, True, NAVY),
        ("ceza_tutari_tl: 10 sınıf (0 TL ve 9 kademe)\nProblem türü: sınıflandırma", 15, False, INK),
        ("Neden önemli", 18, True, NAVY),
        ("2026'da ceza, sınırın kaç km/s aşıldığına göre\n9 kademeye bağlandı: 2.000 TL ile 30.000 TL arası;\nyüksek aşımda ehliyet 30 ile 90 gün geri alınıyor.", 15, False, INK),
        ("Neden model", 18, True, NAVY),
        ("Kural belli; kanun bu projede sınavın cevap anahtarı.\nSoru: model kuralı örneklerden ne kadar öğreniyor?", 15, False, INK),
    ])
    panel(s, Inches(6.85), Inches(1.75), Inches(5.9), Inches(4.95))
    text_box(s, Inches(7.15), Inches(1.95), Inches(5.4), Inches(4.6), [
        ("Veri: kanundan üretildi", 18, True, NAVY),
        ("Kişi bazında ceza kaydı kamuya açık değil.\nCeza: KTK 51/2 (7574 sayılı Kanun)\nEhliyet yaptırımı: KTK 51/3 ve 51/4\nHız sınırları: Karayolları Trafik Yönetmeliği md. 100", 15, False, INK),
        ("Temizlik", 18, True, NAVY),
        (f"{m['rows_raw']} satır → {m['rows_clean']} satır", 26, True, RED),
        (f"Silinen: {m['missing_speed']} boş hız, {dataset.IMPOSSIBLE_SPEED_ROWS} imkânsız hız, {m['duplicates']} tekrar\n"
         f"Düzeltilen: {m['missing_weather']} boş hava durumu, {m['spelling_fixed']} bozuk yazım", 15, False, INK),
        ("Bölme", 18, True, NAVY),
        (f"%75 eğitim ({m['train_rows']} satır), %25 test ({m['test_rows']} satır)", 15, False, INK),
    ])

    # 3. Exploration
    s = deck.slides.add_slide(blank)
    header(s, "Keşifçi veri analizi", "Ceza, sınırın kaç km/s aşıldığına bağlı", 3, total)
    picture(s, FIGURES / "asim_ceza.png", Inches(0.6), Inches(1.75), Inches(7.6),
            "Ceza tutarı, sınırın kaç km/s aşıldığına göre basamak basamak artıyor")
    text_box(s, Inches(8.6), Inches(1.9), Inches(4.4), Inches(5), [
        ("Bulgular", 18, True, NAVY),
        ("Ceza basamak basamak artıyor;\nher basamak kanunda bir km/s aralığı.", 15, False, INK),
        ("Yerleşim yeri içinde basamaklar\ndaha erken başlıyor.", 15, False, INK),
        ("Sınırı yerleşim yeri içinde 5'e, dışında\n10 km/s'e kadar aşan kayıtlar ceza almıyor.", 15, False, INK),
        ("Hava durumu belirleyici değil;\nveri havadan bağımsız üretildi,\nfarklar az kayıtlı gruplarda rastlantı.", 15, False, INK),
    ])

    # 4. Model and results
    s = deck.slides.add_slide(blank)
    header(s, "Model ve sonuç", "Doğru özellik, aynı yöntemi kanuna yaklaştırıyor", 4, total)
    picture(s, FIGURES / "model_karsilastirma.png", Inches(0.5), Inches(1.75), Inches(6.6),
            "Test doğruluğu ve kanunla uyum: Model A ile Model B")
    # Three messages only; the boundary count and the learning-curve detail stay in the report and the README.
    text_box(s, Inches(7.4), Inches(1.75), Inches(5.7), Inches(5.2), [
        ("Karar ağacı, iki model", 18, True, NAVY),
        ("A: ham sütunlar\nB: aşım (km/s) ve yerleşim yeri içi mi", 14, False, INK),
        (f"Kanunla uyum: {thousands(m['grid_rows'])} durumda kanunla\naynı cezanın verildiği durumların oranı", 13, False, MUTED),
        (f"A {pct(raw['law_agreement'])}   ·   B {pct(eng['law_agreement'])}", 24, True, GREEN),
        (f"A eğitim verisini ezberliyor:\neğitim {pct(raw['train_accuracy'])}, test {pct(raw['test_accuracy'])}.", 15, False, INK),
        (f"B'nin kanundan ayrıldığı {m['model_b_disagreements']} durumun hepsi\n"
         "eğitimde görmediği girdilerde,\nkademe sınırlarının hemen yanında.", 15, False, INK),
        (f"Görmediği girdilerde uyum {pct(m['unseen_agreement'])};\neğitim verisi büyüdükçe genelde yükseliyor.", 15, False, INK),
    ])

    # 5. Mini app and takeaway
    s = deck.slides.add_slide(blank)
    header(s, "Mini uygulama ve çıkarım", "Girdi → tahmin → anlamlı cümle", 5, total)
    panel(s, Inches(0.6), Inches(1.65), Inches(12.1), Inches(3.05), RGBColor(0x1B, 0x26, 0x33)).adjustments[0] = 0.03
    label, light, green = RGBColor(0x9F, 0xB3, 0xC8), RGBColor(0xE6, 0xEC, 0xF2), RGBColor(0x8F, 0xD3, 0xA8)
    session = sample_session()
    situation, answer = output_lines(session[3])
    warn_situation, warn_answer = output_lines(m["examples"][5])
    text_box(s, Inches(0.9), Inches(1.8), Inches(11.6), Inches(3.1), [
        ("Örnek çalıştırma (Aşama 6 çıktısı; seçenek listeleri [...] ile kısaltıldı)", 11, False, label),
        ("\n".join(session[:3]), 13, False, light),
        (situation, 13, True, green),
        (answer, 13, False, light),
        ("Model kademe sınırının yakınında yanıldığında (Aşama 5 örneği)", 11, False, label),
        (warn_situation, 13, True, green),
        (warn_answer, 13, False, light),
    ])
    text_box(s, Inches(0.6), Inches(5.15), Inches(12.1), Inches(1.7), [
        ("Çıkarım", 18, True, NAVY),
        ("Aynı algoritma, aldığı bilgiye göre ya eğitim verisini ezberliyor ya da kanunun kuralına yaklaşıyor.\nModel sınırın yakınında yanılırsa uygulama bunu kanun tablosuyla yakalayıp söylüyor.", 15, False, INK),
        ("Zayıf yönler: veri kanundan üretildi; ölçüm hatası ve yerel levha sınırları yok.\nTutarların 2027'den itibaren yeniden değerlemeyle artması bekleniyor; bunu doğrulayan resmî duyuru bulamadım.", 13, False, MUTED),
    ])

    problems = overflowing_lines(deck)
    if problems:
        raise SystemExit("lines that would wrap on their own:\n" + "\n".join(problems))
    set_document_properties(deck)
    path.parent.mkdir(exist_ok=True)
    deck.save(path)
    clean_package(path, len(deck.slides))
    return path


def main() -> None:
    metrics = load_metrics()
    if not EDGE.exists():
        raise SystemExit(f"Edge not found at {EDGE}")
    print("wrote", build_report(metrics))
    print("wrote", build_slides(metrics))


if __name__ == "__main__":
    main()
