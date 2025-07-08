# File: src/metrics/main.py

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import argparse
import json
import logging
from pathlib import Path

from metrics.ast_metrics.extractor import ASTMetricExtractor
from metrics.ast_metrics.gather import gather_ast_metrics


def analyse_file(file_path: Path, raw: bool = False, out_path: Path = None):
    """Analyse a single file and optionally write output to a file."""
    if not file_path.is_file():
        logging.error(f"File not found: {file_path}")
        return

    if raw:
        results = gather_ast_metrics(str(file_path))
    else:
        extractor = ASTMetricExtractor(str(file_path))
        results = extractor.extract()

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logging.info(f"Output written to {out_path}")
    else:
        print(json.dumps(results, indent=2))


def analyse_directory(directory: Path, raw: bool = False):
    """Recursively analyse all Python files in a directory."""
    if not directory.is_dir():
        logging.error(f"Not a directory: {directory}")
        return

    py_files = list(directory.rglob("*.py"))
    if not py_files:
        logging.warning(f"No Python files found in: {directory}")
        return

    for file in py_files:
        print(f"\n{'=' * 60}\n📄 {file}")
        analyse_file(file, raw=raw)


def main():
    parser = argparse.ArgumentParser(description="Plugin-based AST metrics extractor")
    parser.add_argument("--file", "-f", type=str, help="Path to a Python file to analyse")
    parser.add_argument("--dir", "-d", type=str, help="Path to a directory to recursively analyse")
    parser.add_argument("--raw", action="store_true", help="Output raw list instead of full metric dict")
    parser.add_argument("--out", type=str, help="Path to save output JSON file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if args.file:
        analyse_file(Path(args.file), raw=args.raw, out_path=Path(args.out) if args.out else None)
    elif args.dir:
        analyse_directory(Path(args.dir), raw=args.raw)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
