import argparse
import os
import sys
from pdf_processor.processor import PDFPostProcessor


def parse_args(args=None):
    """
    Parses command line arguments for the PDF post-processor tool.
    """
    parser = argparse.ArgumentParser(
        description="PDF Post-Processor: Randomly converts body paragraph words into image snippets in place."
    )
    
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Path to the input PDF file."
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path for the output processed PDF file. Defaults to <input_stem>_processed.pdf."
    )

    parser.add_argument(
        "-p", "--probability",
        type=float,
        default=0.15,
        help="Probability ratio (0.0 to 1.0) of converting body words into images. Default: 0.15."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Integer seed for random word selection (ensures reproducible output)."
    )

    parser.add_argument(
        "--min-word-len",
        type=int,
        default=3,
        help="Minimum character length of words to consider for image replacement. Default: 3."
    )

    parser.add_argument(
        "--dpi-scale",
        type=float,
        default=3.0,
        help="DPI resolution scale multiplier for text-to-image rendering. Default: 3.0."
    )

    parser.add_argument(
        "--font-path",
        type=str,
        default=None,
        help="Optional path to a custom TrueType (.ttf/.otf) font file."
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output during processing."
    )

    return parser.parse_args(args)


def main():
    """
    CLI entrypoint execution function.
    """
    args = parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        print(f"Error: Input PDF file not found at '{input_path}'", file=sys.stderr)
        sys.exit(1)

    if not args.output:
        stem, ext = os.path.splitext(input_path)
        output_path = f"{stem}_processed{ext}"
    else:
        output_path = os.path.abspath(args.output)

    if args.verbose:
        print("=== PDF Post-Processor CLI ===")
        print(f"Input PDF:        {input_path}")
        print(f"Output PDF:       {output_path}")
        print(f"Replacement Prob: {args.probability}")
        print(f"Random Seed:      {args.seed}")
        print(f"Min Word Length:  {args.min_word_len}")
        print(f"DPI Scale:        {args.dpi_scale}")
        print("==============================")

    try:
        processor = PDFPostProcessor(
            input_path=input_path,
            output_path=output_path,
            probability=args.probability,
            seed=args.seed,
            min_word_len=args.min_word_len,
            dpi_scale=args.dpi_scale,
            font_path=args.font_path,
            verbose=args.verbose
        )
        
        summary = processor.process()
        
        print("\nProcessing completed successfully!")
        print(f"Total Pages:           {summary['total_pages']}")
        print(f"Body Words Scanned:    {summary['total_words_processed']}")
        print(f"Words Converted:       {summary['total_words_replaced']}")
        print(f"Output File Saved:     {summary['output_path']}")

    except Exception as e:
        print(f"\nError processing PDF document: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
