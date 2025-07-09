import sys
import argparse
import json
import logging
from pathlib import Path

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

    # 🧠 ML Signal: Core metric extraction logic
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

    # ✅ Best Practice: Export JSON
    if format in ("json", "both"):
        json_path = json_out or out_path or Path("metrics.json")
        try:
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result_dict, f, indent=2)
            logging.info(f"📄 JSON saved: {json_path}")
        except Exception as e:
            logging.error(f"❌ Failed to write JSON: {e}")

    # ✅ Best Practice: Export CSV
    if format in ("csv", "both"):
        csv_path = csv_out or (Path(out_path or "metrics.csv").with_suffix(".csv"))
        try:
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("File," + ",".join(metric_names) + "\n")
                row = [result_dict.get(name, 0) for name in metric_names]
                f.write(f"{file_path.name}," + ",".join(map(str, row)) + "\n")
            logging.info(f"📄 CSV saved: {csv_path}")
        except Exception as e:
            logging.error(f"❌ Failed to write CSV: {e}")

    # ✅ Markdown-compatible summary output
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

    # ⚠️ SAST Risk: Validate fail-on logic for scoring thresholds
    if fail_threshold is not None:
        if fail_on == "ast":
            metric_keys = [k for k in result_dict if not k.startswith("number_of_")]
        elif fail_on == "bandit":
            metric_keys = [k for k in result_dict if k.startswith("number_of_") and "security" in k]
        elif fail_on == "flake8":
            metric_keys = [k for k in result_dict if "styling" in k or "line_length" in k]
        elif fail_on == "cloc":
            metric_keys = [k for k in result_dict if "line" in k or "comment" in k]
        else:
            metric_keys = result_dict.keys()

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
        description="🔍 Code Analyser - AST, Bandit, Cloc, Flake8 plugin-based metric extractor"
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
    parser.add_argument("--fail-threshold", type=int, help="Exit 1 if metric sum exceeds this threshold")
    parser.add_argument("--fail-on", choices=["all", "ast", "bandit", "flake8", "cloc"], default="all", help="Restrict threshold to a metric type")
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
