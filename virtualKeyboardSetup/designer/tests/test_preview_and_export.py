"""
Unit Tests for Preview and Export Services.
Tests text wrapping, truncation with '..' suffix, and preview/pdf rendering utilities.
"""

import unittest
from services.preview_service import PreviewService
from services.pdf_exporter import PdfExporter


class TestPreviewAndExportServices(unittest.TestCase):
    def test_wrap_and_truncate_short_text(self):
        text = "Short text"
        # Mock width_fn returning len(s) * 5
        width_fn = lambda s: len(s) * 5.0
        lines = PreviewService._wrap_and_truncate_text(
            text, max_w=100.0, max_h=50.0, line_height=10.0, width_fn=width_fn
        )
        self.assertEqual(lines, ["Short text"])

    def test_wrap_and_truncate_long_text_adds_double_dots(self):
        text = "This is a very long text string that will definitely overflow the button height"
        width_fn = lambda s: len(s) * 5.0
        # Allow max 2 lines (max_h = 20.0, line_height = 10.0)
        lines = PreviewService._wrap_and_truncate_text(
            text, max_w=80.0, max_h=20.0, line_height=10.0, width_fn=width_fn
        )
        self.assertLessEqual(len(lines), 2)
        self.assertTrue(lines[-1].endswith(".."))

    def test_pdf_exporter_wrap_and_truncate_adds_double_dots(self):
        text = "Supercalifragilisticexpialidocious text that goes on forever"
        string_width_fn = lambda s: len(s) * 6.0
        lines = PdfExporter._wrap_and_truncate_text(
            text, max_w=70.0, max_h=25.0, line_height=12.0, string_width_fn=string_width_fn
        )
        self.assertLessEqual(len(lines), 2)
        self.assertTrue(lines[-1].endswith(".."))


if __name__ == "__main__":
    unittest.main()
