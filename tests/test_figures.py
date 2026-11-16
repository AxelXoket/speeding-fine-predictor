"""Saved charts and the report are what a reader sees: no text may run off a chart's edge, the report is one page."""
import io
import pathlib
import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402
from pypdf import PdfReader  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
BORDER_PX = 3
INK_LEVEL = 200  # a pixel darker than this (0-255 grey) is ink, not background


def ink_on_border(image: Image.Image) -> list[str]:
    """Edges of the image that carry ink; text cut off at the edge always leaves ink there."""
    grey = image.convert("L")
    width, height = grey.size
    bands = {
        "left": (0, 0, BORDER_PX, height),
        "right": (width - BORDER_PX, 0, width, height),
        "top": (0, 0, width, BORDER_PX),
        "bottom": (0, height - BORDER_PX, width, height),
    }
    return [side for side, box in bands.items() if grey.crop(box).getextrema()[0] < INK_LEVEL]


def chart_png(title: str) -> Image.Image:
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot([0, 1], [0, 1])
    ax.set_title(title)
    buffer = io.BytesIO()
    fig.savefig(buffer, dpi=100)
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer)


class ChartsAreNotCutOff(unittest.TestCase):
    def test_every_saved_chart_has_a_clean_border(self):
        charts = sorted((ROOT / "figures").glob("*.png"))
        self.assertGreaterEqual(len(charts), 5)
        cut = {path.name: ink_on_border(Image.open(path)) for path in charts}
        self.assertEqual({name: sides for name, sides in cut.items() if sides}, {})

    def test_detector_flags_a_title_wider_than_the_chart(self):
        self.assertEqual(ink_on_border(chart_png("Kısa başlık")), [])
        self.assertIn("right", ink_on_border(chart_png("Bu başlık grafiğin genişliğine asla sığmayacak kadar uzun bir başlık")))


class ReportIsOnePage(unittest.TestCase):
    def test_committed_report_has_exactly_one_page(self):
        self.assertEqual(len(PdfReader(ROOT / "docs" / "rapor.pdf").pages), 1)


if __name__ == "__main__":
    unittest.main()
