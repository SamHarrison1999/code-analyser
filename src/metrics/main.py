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
    include_bandit: bool = False,
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
        logging.error(f"File not found: {file_path}")
        return 1

    # ─── Extract Metrics ───────────────────────────────────────────────
    values = gather_all_metrics(str(file_path))
    result_dict = dict(zip(get_all_metric_names(), values))

    # ─── Write Outputs ─────────────────────────────────────────────────
    if format in ("json", "both"):
        json_path = json_out or out_path or Path("metrics.json")
        try:
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result_dict, f, indent=2)
            logging.info(f"[CLI] JSON output written to: {json_path}")
        except Exception as e:
            logging.error(f"[CLI] Failed to write JSON: {e}")

    if format in ("csv", "both"):
        headers = get_all_metric_names()
        csv_path = csv_out or Path(out_path or "metrics.csv").with_suffix(".csv")
        try:
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("File," + ",".join(headers) + "\n")
                row = [result_dict.get(h, 0) for h in headers]
                f.write(f"{file_path.name}," + ",".join(map(str, row)) + "\n")
            logging.info(f"[CLI] CSV output written to: {csv_path}")
        except Exception as e:
            logging.error(f"[CLI] Failed to write CSV: {e}")

    # ─── Optional Summary Output ───────────────────────────────────────
    if show_summary or markdown_summary or save_summary_txt:
        summary = format_summary_table(result_dict)
        if show_summary or markdown_summary:
            logging.info("[CLI Summary Table]\n" + summary)
        if save_summary_txt:
            try:
                save_summary_txt.parent.mkdir(parents=True, exist_ok=True)
                with open(save_summary_txt, "w", encoding="utf-8") as f:
                    f.write(summary + "\n")
                logging.info(f"[CLI] Markdown summary saved to: {save_summary_txt}")
            except Exception as e:
                logging.error(f"[CLI] Failed to save summary: {e}")

    # ─── Failure Exit Checks ──────────────────────────────────────────
    if fail_threshold is not None:
        metric_keys = result_dict.keys()
        if fail_on == "ast":
            metric_keys = [k for k in result_dict if not k.startswith("number_of_")]
        elif fail_on == "bandit":
            metric_keys = [k for k in result_dict if k.startswith("number_of_")]
        total = sum(result_dict.get(k, 0) for k in metric_keys)
        if total > fail_threshold:
            logging.warning(
                f"[CLI] Total {fail_on.upper()} metric sum {total} exceeds threshold {fail_threshold}."
            )
            return 1

    return 0


def format_summary_table(metrics: dict) -> str:
    """Generate a markdown-style table for summary reporting."""
    nonzero = [(k, v) for k, v in metrics.items() if v > 0]
    if not nonzero:
        return "No non-zero metrics recorded."
    lines = ["Metric | Value", "-------|------"]
    lines += [f"{k} | {v}" for k, v in nonzero]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Plugin-based code metric extractor")
    parser.add_argument("--file", "-f", type=str, help="Path to a Python file to analyse")
    parser.add_argument("--dir", "-d", type=str, help="Directory to recursively analyse")
    parser.add_argument("--raw", action="store_true", help="Return raw list instead of dict")
    parser.add_argument("--format", choices=["json", "csv", "both"], default="json", help="Output format")
    parser.add_argument("--out", type=str, help="Base path for output files")
    parser.add_argument("--json-out", type=str, help="Optional path for JSON output")
    parser.add_argument("--csv-out", type=str, help="Optional path for CSV output")
    parser.add_argument("--bandit", action="store_true", help="(legacy, no effect) Bandit always included now")
    parser.add_argument("--summary", action="store_true", help="Print metric summary to stdout")
    parser.add_argument("--metrics-summary-table", action="store_true", help="Alias for --summary")
    parser.add_argument("--save-summary-txt", type=str, help="Save summary table to text file")
    parser.add_argument("--fail-threshold", type=int, help="Exit 1 if total metric sum exceeds this")
    parser.add_argument("--fail-on", choices=["all", "ast", "bandit"], default="all", help="Restrict --fail-threshold scope")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s"
    )

    if args.file:
        code = analyse_file(
            Path(args.file),
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
        for py_file in Path(args.dir).rglob("*.py"):
            logging.info(f"\n{'=' * 60}\n📄 {py_file}")
            result = analyse_file(Path(py_file), raw=args.raw, format=args.format)
            if result != 0:
                failed += 1
        sys.exit(failed)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
