import sys
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

from metrics.gather import gather_all_metrics, get_all_metric_names


def analyse_file(
    file_path: Path,
    raw: bool = False,
    out_path: Path = None,
    format: str = "json",
    json_out: Path = None,
    csv_out: Path = None,
    fail_threshold: int = None,
    fail_on: str = "all",
    show_summary: bool = False,
    save_summary_txt: Path = None,
    markdown_summary: bool = False,
) -> int:
    """Analyse a single file and optionally write output."""
    if not file_path.is_file():
        logging.error(f"❌ File not found: {file_path}")
        return 1

    try:
        values = gather_all_metrics(str(file_path))
        metric_names = get_all_metric_names()
        result_dict = dict(zip(metric_names, values))
    except Exception as e:
        logging.error(f"❌ Metric extraction failed: {e}")
        return 1

    if raw:
        print(values)
        return 0

    wrapped_json = {
        "file": str(file_path),
        "timestamp": datetime.utcnow().isoformat(),
        "metrics": result_dict
    }

    if format in ("json", "both"):
        json_path = json_out or out_path or Path("metrics.json")
        try:
            json_path = json_path.with_suffix(".json")
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps(wrapped_json, indent=2), encoding="utf-8")
            logging.info(f"📄 JSON saved: {json_path}")
        except Exception as e:
            logging.error(f"❌ Failed to write JSON: {e}")

    if format in ("csv", "both"):
        csv_path = csv_out or (out_path or Path("metrics.csv")).with_suffix(".csv")
        try:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("File," + ",".join(metric_names) + "\n")
                row = [result_dict.get(name, 0) for name in metric_names]
                f.write(f"{file_path.name}," + ",".join(map(str, row)) + "\n")
            logging.info(f"📄 CSV saved: {csv_path}")
        except Exception as e:
            logging.error(f"❌ Failed to write CSV: {e}")

    if show_summary or markdown_summary or save_summary_txt:
        summary = format_summary_table(result_dict)
        if show_summary or markdown_summary:
            print("\n📊 Summary:\n" + summary)
        if save_summary_txt:
            try:
                save_summary_txt.parent.mkdir(parents=True, exist_ok=True)
                save_summary_txt.write_text(summary + "\n", encoding="utf-8")
                logging.info(f"📝 Markdown summary saved: {save_summary_txt}")
            except Exception as e:
                logging.error(f"❌ Failed to save summary: {e}")

    if fail_threshold is not None:
        if fail_on == "ast":
            ast_keys = get_all_metric_names()[:14]
            metric_keys = [k for k in result_dict if k in ast_keys]
        elif fail_on == "bandit":
            metric_keys = [k for k in result_dict if "bandit" in k or "security" in k]
        elif fail_on == "flake8":
            metric_keys = [k for k in result_dict if "flake8" in k or "line_length" in k]
        elif fail_on == "cloc":
            metric_keys = [k for k in result_dict if "line" in k or "comment" in k]
        elif fail_on == "lizard":
            metric_keys = [k for k in result_dict if any(p in k for p in [
                "complexity", "token", "parameter", "maintainability"
            ])]
        elif fail_on == "pydocstyle":
            metric_keys = [k for k in result_dict if "docstring" in k or "pydocstyle" in k]
        elif fail_on == "pyflakes":
            metric_keys = [k for k in result_dict if "undefined" in k or "syntax" in k]
        elif fail_on == "pylint":
            from metrics.pylint_metrics.gather import get_pylint_metric_names
            pylint_keys = set(get_pylint_metric_names())
            metric_keys = [k for k in result_dict if k in pylint_keys]
        elif fail_on == "radon":
            metric_keys = [k for k in result_dict if any(sub in k for sub in [
                "halstead", "logical_lines", "blank_lines", "doc_strings"
            ])]
        else:
            metric_keys = list(result_dict.keys())

        total = sum(result_dict.get(k, 0) or 0 for k in metric_keys)
        if total > fail_threshold:
            logging.warning(f"⚠️ Total {fail_on.upper()} metrics = {total} > threshold = {fail_threshold}")
            return 1

    return 0


def format_summary_table(metrics: dict) -> str:
    """Render non-zero metrics as a markdown-style table."""
    nonzero = [(k, v) for k, v in metrics.items() if v]
    if not nonzero:
        return "No non-zero metrics recorded."
    lines = ["Metric | Value", "-------|------"]
    lines += [f"{k} | {v}" for k, v in nonzero]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="🔍 Code Analyser — AST, Bandit, Cloc, Flake8, Lizard, Pydocstyle, Pyflakes, Pylint, Radon metric extractor"
    )
    parser.add_argument("--file", "-f", type=str, help="Analyse a single Python file")
    parser.add_argument("--dir", "-d", type=str, help="Recursively analyse a directory of Python files")
    parser.add_argument("--raw", action="store_true", help="Return raw list instead of named dict")
    parser.add_argument("--format", choices=["json", "csv", "both"], default="json", help="Output format")
    parser.add_argument("--out", type=str, help="Base path for output files")
    parser.add_argument("--json-out", type=str, help="Path to write JSON")
    parser.add_argument("--csv-out", type=str, help="Path to write CSV")
    parser.add_argument("--summary", action="store_true", help="Print metric summary to terminal")
    parser.add_argument("--metrics-summary-table", action="store_true", help="Alias for --summary")
    parser.add_argument("--save-summary-txt", type=str, help="Save summary as markdown table")
    parser.add_argument(
        "--fail-threshold", type=int,
        help="Exit 1 if metric sum exceeds this threshold"
    )
    parser.add_argument(
        "--fail-on",
        choices=["all", "ast", "bandit", "flake8", "cloc", "lizard", "pydocstyle", "pyflakes", "pylint", "radon"],
        default="all",
        help="Restrict threshold check to a specific metric group"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s"
    )

    if args.file:
        code = analyse_file(
            file_path=Path(args.file),
            raw=args.raw,
            out_path=Path(args.out) if args.out else None,
            format=args.format,
            json_out=Path(args.json_out) if args.json_out else None,
            csv_out=Path(args.csv_out) if args.csv_out else None,
            fail_threshold=args.fail_threshold,
            fail_on=args.fail_on,
            show_summary=args.summary or args.metrics_summary_table,
            save_summary_txt=Path(args.save_summary_txt) if args.save_summary_txt else None,
            markdown_summary=args.metrics_summary_table,
        )
        sys.exit(code)

    elif args.dir:
        failed = 0
        for file in Path(args.dir).rglob("*.py"):
            logging.info(f"\n📄 {file}")
            result = analyse_file(file, raw=args.raw, format=args.format)
            if result != 0:
                failed += 1
        sys.exit(failed)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
