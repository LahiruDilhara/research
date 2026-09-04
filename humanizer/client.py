#!/usr/bin/env python3
"""
Humanizer CLI Client
Sends text to the local Humanizer Playwright service and outputs the result.
"""

import argparse
import json
import sys
import httpx

DEFAULT_URL = "http://127.0.0.1:8000/humanize"

def main():
    parser = argparse.ArgumentParser(
        description="Send text to the local Humanizer service and print the humanized result."
    )
    parser.add_argument(
        "text",
        nargs="?",
        default=None,
        help="The text string to humanize (optional if reading from stdin or --file)."
    )
    parser.add_argument(
        "-f", "--file",
        type=str,
        default=None,
        help="Path to a text file to read input from."
    )
    parser.add_argument(
        "-o", "--out",
        type=str,
        default=None,
        help="Path to an output file to save the humanized text to (if omitted, prints to stdout)."
    )
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_URL,
        help=f"Humanizer service endpoint URL (default: {DEFAULT_URL})"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the raw JSON response instead of plain text."
    )
    parser.add_argument(
        "--style",
        type=str,
        default="standard",
        help="Rewriting style (default: standard)"
    )

    args = parser.parse_args()

    # Determine input text
    input_text = None
    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                input_text = f.read()
        except Exception as e:
            print(f"Error reading input file {args.file}: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.text:
        input_text = args.text
    elif not sys.stdin.isatty():
        input_text = sys.stdin.read()

    if not input_text or not input_text.strip():
        print("Error: No input text provided.", file=sys.stderr)
        print("Usage examples:", file=sys.stderr)
        print("  python client.py 'Your text to humanize'", file=sys.stderr)
        print("  python client.py -f input.txt -o output.txt", file=sys.stderr)
        print("  cat input.txt | python client.py", file=sys.stderr)
        sys.exit(1)

    payload = {
        "text": input_text.strip(),
        "style": args.style
    }

    try:
        response = httpx.post(args.url, json=payload, timeout=180.0)
    except httpx.ConnectError:
        print(f"Error: Could not connect to Humanizer service at {args.url}", file=sys.stderr)
        print("Ensure the server is running via: uv run python main.py", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Request failed: {e}", file=sys.stderr)
        sys.exit(1)

    if response.status_code != 200:
        print(f"Server error (status {response.status_code}): {response.text}", file=sys.stderr)
        sys.exit(1)

    data = response.json()
    
    if not data.get("success"):
        print(f"Humanizer error: {data.get("error")}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        output_content = json.dumps(data, indent=2)
    else:
        output_content = data.get("text") or data.get("humanized_text") or ""

    if args.out:
        try:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(output_content)
            print(f"Successfully saved humanized text to {args.out}", file=sys.stderr)
        except Exception as e:
            print(f"Error writing to output file {args.out}: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(output_content)

if __name__ == "__main__":
    main()
