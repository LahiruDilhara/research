import argparse
import os
import sys
from pdf_processor.processor import PDFPostProcessor


def parse_args(args=None):
    """
    Parses command line arguments for the PDF post-processor tool with parallel page processing controls.
    """
    parser = argparse.ArgumentParser(
        description="PDF Post-Processor: 4-stage pipeline converting body words to images, homoglyphs, zero-width chars, and layout disruptions."
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
        help="Stage 1: Probability ratio (0.0 to 1.0) of converting body words into images. Default: 0.15."
    )

    parser.add_argument(
        "--homo-prob",
        type=float,
        default=0.15,
        help="Stage 2: Probability ratio (0.0 to 1.0) of replacing Latin letters with homoglyphs. Default: 0.15."
    )

    parser.add_argument(
        "--zw-prob",
        type=float,
        default=0.15,
        help="Stage 3: Probability ratio (0.0 to 1.0) of injecting zero-width characters. Default: 0.15."
    )

    parser.add_argument(
        "--disrupt-prob",
        type=float,
        default=0.15,
        help="Stage 4: Probability ratio (0.0 to 1.0) of layout disruption (scrambled text layer). Default: 0.15."
    )

    parser.add_argument(
        "--max-images-per-page",
        type=int,
        default=0,
        help="Stage 1: Maximum number of word images allowed per page. (0 = unlimited). Default: 0."
    )

    parser.add_argument(
        "--max-images-per-para",
        type=int,
        default=0,
        help="Stage 1: Maximum number of word images allowed per body paragraph. (0 = unlimited). Default: 0."
    )

    parser.add_argument(
        "--zw-count",
        type=int,
        default=2,
        help="Stage 3: Number of invisible zero-width characters injected per selected word. Default: 2."
    )

    parser.add_argument(
        "--disrupt-multiplier",
        type=float,
        default=1.5,
        help="Stage 4: Length multiplier for scrambled invisible disruption overlay text. Default: 1.5."
    )

    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes for page processing. (0 = auto-detect CPU cores, 1 = sequential). Default: 1."
    )

    parser.add_argument(
        "--stage",
        type=str,
        choices=["all", "stage1", "stage2", "stage3", "stage4"],
        default="all",
        help="Pipeline stage execution mode: 'all' (Stages 1-4), 'stage1' (Images), 'stage2' (Homoglyphs), 'stage3' (Zero-width), 'stage4' (Disruption). Default: 'all'."
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
        help="Minimum character length of words to consider. Default: 3."
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
        print(f"Input PDF:            {input_path}")
        print(f"Output PDF:           {output_path}")
        print(f"Execution Stage:      {args.stage}")
        print(f"Parallel Workers:     {args.workers if args.workers > 0 else f'Auto ({os.cpu_count()} CPU cores)'}")
        print(f"Stage 1 Image Prob:   {args.probability}")
        print(f"Max Images / Page:    {args.max_images_per_page if args.max_images_per_page > 0 else 'Unlimited'}")
        print(f"Max Images / Para:    {args.max_images_per_para if args.max_images_per_para > 0 else 'Unlimited'}")
        print(f"Stage 2 Homoglyph Prob:{args.homo_prob}")
        print(f"Stage 3 ZW-Char Prob: {args.zw_prob} (Count: {args.zw_count} per word)")
        print(f"Stage 4 Disruption Prob:{args.disrupt_prob} (Multiplier: {args.disrupt_multiplier}x)")
        print(f"Random Seed:          {args.seed}")
        print(f"Min Word Length:      {args.min_word_len}")
        print(f"DPI Scale:            {args.dpi_scale}")
        print("==============================")

    try:
        processor = PDFPostProcessor(
            input_path=input_path,
            output_path=output_path,
            probability=args.probability,
            homo_probability=args.homo_prob,
            zw_probability=args.zw_prob,
            disrupt_probability=args.disrupt_prob,
            stage=args.stage,
            seed=args.seed,
            min_word_len=args.min_word_len,
            dpi_scale=args.dpi_scale,
            max_images_per_page=args.max_images_per_page,
            max_images_per_para=args.max_images_per_para,
            zw_count=args.zw_count,
            disrupt_multiplier=args.disrupt_multiplier,
            workers=args.workers,
            font_path=args.font_path,
            verbose=args.verbose
        )
        
        summary = processor.process()
        
        print("\nProcessing completed successfully!")
        print(f"Total Pages:               {summary['total_pages']}")
        print(f"Body Words Scanned:        {summary['total_words_processed']}")
        print(f"Stage 1 Images Replaced:   {summary['total_image_replacements']}")
        print(f"Stage 2 Homoglyphs Swapped:{summary['total_homo_substitutions']}")
        print(f"Stage 3 ZW-Chars Injected: {summary['total_zw_injections']}")
        print(f"Stage 4 Layout Disruptions:{summary['total_layout_disruptions']}")
        print(f"Output File Saved:         {summary['output_path']}")

    except Exception as e:
        print(f"\nError processing PDF document: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
