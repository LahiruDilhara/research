import sys, os, time
sys.path.insert(0, os.path.abspath("."))
import pymupdf as fitz
from pdf_processor.processor import PDFPostProcessor


def test_parallel():
    input_path = "input_sample.pdf"
    output_seq = "tmp_seq.pdf"
    output_par = "tmp_par.pdf"

    print("--- Running Sequential (workers=1) ---")
    t0 = time.time()
    proc1 = PDFPostProcessor(input_path, output_seq, workers=1, seed=42)
    s1 = proc1.process()
    t1 = time.time() - t0
    print(f"Sequential Time: {t1:.3f}s | Processed Words: {s1['total_words_processed']}")

    print("\n--- Running Parallel (workers=2) ---")
    t0 = time.time()
    proc2 = PDFPostProcessor(input_path, output_par, workers=2, seed=42)
    s2 = proc2.process()
    t2 = time.time() - t0
    print(f"Parallel Time:   {t2:.3f}s | Processed Words: {s2['total_words_processed']}")

    if os.path.exists(output_seq): os.remove(output_seq)
    if os.path.exists(output_par): os.remove(output_par)


if __name__ == "__main__":
    test_parallel()
