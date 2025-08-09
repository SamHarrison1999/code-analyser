"""
main.py — CLI driver for the Code Analyser

Supports modular extraction across AST, Bandit, Cloc, Flake8, Lizard, Pydocstyle,
Pyflakes, Pylint, Radon, Vulture, and SonarQube via plugin-based backends.

Features:
- JSON, CSV, HTML metric export
- Token heatmap export
- Together.ai annotation integration
- Batch folder annotation
- Markdown and terminal summary output
- Torch/TF overlay toggle
- GUI launch
"""

import sys
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Union

from metrics.gather import gather_all_metrics
from ml.ollama_llm_annotator import annotate_code_with_together_ai
from ml.export_helpers import export_html_dashboard, export_token_heatmap_csv
from gui import launch_gui

logger = logging.getLogger(__name__)


def analyse_file(
    file_path: Path,
    raw: bool = False,
    out_path: Union[Path, None] = None,
    format: str = "json",
    json_out: Union[Path, None] = None,
    csv_out: Union[Path, None] = None,
    fail_threshold: Union[int, None] = None,
    fail_on: str = "all",
    show_summary: bool = False,
    save_summary_txt: Union[Path, None] = None,
    markdown_summary: bool = False,
    export_dir: Union[Path, None] = None,
    export_html: bool = False,
    export_heatmap_csv: bool = False,
    use_torch_overlay: bool = False,
) -> int:
    """Analyse a single file and optionally write outputs including overlays."""
    if not file_path.is_file():
        logger.error(f"❌ File not found: {file_path}")
        return 1

    try:
        result_dict = gather_all_metrics(str(file_path))
        if not isinstance(result_dict, dict):
            raise ValueError("gather_all_metrics did not return a dictionary")
        metric_names = list(result_dict.keys())
    except Exception as e:
        logger.error(f"❌ Metric extraction failed: {e}")
        return 1

    if raw:
        print(result_dict)
        return 0

    wrapped_json = {
        "file": str(file_path),
        "timestamp": datetime.now().isoformat(),
        "metrics": result_dict,
        "overlay_model": "torch" if use_torch_overlay else "tensorflow",
    }

    export_path = export_dir or Path.cwd()

    if format in ("json", "both"):
        try:
            json_path = (json_out or export_path / "metrics.json").with_suffix(".json")
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps(wrapped_json, indent=2), encoding="utf-8")
            logger.info(f"📄 JSON saved: {json_path}")
        except Exception as e:
            logger.error(f"❌ Failed to write JSON: {e}")

    if format in ("csv", "both"):
        try:
            csv_path = (csv_out or export_path / "metrics.csv").with_suffix(".csv")
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("File," + ",".join(metric_names) + "\n")
                row = [result_dict.get(name, 0) for name in metric_names]
                f.write(f"{file_path.name}," + ",".join(map(str, row)) + "\n")
            logger.info(f"📄 CSV saved: {csv_path}")
        except Exception as e:
            logger.error(f"❌ Failed to write CSV: {e}")

    if export_html:
        export_html_dashboard(result_dict, export_path)

    if export_heatmap_csv:
        export_token_heatmap_csv(file_path, export_path)

    if show_summary or markdown_summary or save_summary_txt:
        summary = format_summary_table(result_dict)
        if show_summary or markdown_summary:
            print("\n📊 Summary:\n" + summary)
        if save_summary_txt:
            try:
                save_summary_txt.parent.mkdir(parents=True, exist_ok=True)
                save_summary_txt.write_text(summary + "\n", encoding="utf-8")
                logger.info(f"📝 Markdown summary saved: {save_summary_txt}")
            except Exception as e:
                logger.error(f"❌ Failed to save summary: {e}")

    if fail_threshold is not None:
        fail_metric_keys = _select_fail_metrics(fail_on, result_dict)
        total = sum(float(result_dict.get(k, 0) or 0) for k in fail_metric_keys)
        if total > fail_threshold:
            logger.warning(f"⚠️ Total '{fail_on}' metrics = {total} > threshold = {fail_threshold}")
            return 1

    return 0


def _select_fail_metrics(group: str, result_dict: dict) -> list[str]:
    """Select metric keys based on the fail-on group."""
    group_keywords = {
        "ast": ["ast_"],
        "bandit": ["security_", "cwe_"],
        "flake8": ["style", "line_length", "whitespace"],
        "cloc": ["line", "comment"],
        "lizard": ["complexity", "token", "parameter", "function", "maintainability"],
        "pydocstyle": ["docstring", "compliance"],
        "pyflakes": ["undefined", "syntax", "import"],
        "pylint": ["convention", "refactor", "warning", "error", "fatal"],
        "radon": ["halstead", "logical_lines", "blank_lines", "docstring_lines"],
        "vulture": ["unused_"],
        "sonar": [
            "coverage",
            "ncloc",
            "duplicated_",
            "cognitive_",
            "sqale_",
            "bugs",
            "smells",
            "security_",
            "tests",
            "comment_lines_density",
            "classes",
            "files",
        ],
    }

    if group in group_keywords:
        return [k for k in result_dict if any(kw in k for kw in group_keywords[group])]
    return list(result_dict.keys())


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
        description="🔍 Code Analyser CLI — Static + AI + GUI integration"
    )
    parser.add_argument("--file", "-f", type=str, help="Analyse a single Python file")
    parser.add_argument("--dir", "-d", type=str, help="Recursively analyse a directory")
    parser.add_argument(
        "--batch", action="store_true", help="Batch annotate directory before analysis"
    )
    parser.add_argument(
        "--annotate-file-before",
        action="store_true",
        help="🧠 Annotate file before analysis",
    )
    parser.add_argument("--raw", action="store_true", help="Print raw dictionary")
    parser.add_argument(
        "--format",
        choices=["json", "csv", "both"],
        default="json",
        help="Output format",
    )
    parser.add_argument("--json-out", type=str, help="Explicit JSON path")
    parser.add_argument("--csv-out", type=str, help="Explicit CSV path")
    parser.add_argument("--summary", action="store_true", help="Print terminal summary")
    parser.add_argument("--metrics-summary-table", action="store_true", help="Alias for --summary")
    parser.add_argument("--save-summary-txt", type=str, help="Save markdown summary")
    parser.add_argument("--fail-threshold", type=int, help="Fail if score exceeds threshold")
    parser.add_argument(
        "--fail-on",
        choices=[
            "all",
            "ast",
            "bandit",
            "flake8",
            "cloc",
            "lizard",
            "pydocstyle",
            "pyflakes",
            "pylint",
            "radon",
            "vulture",
            "sonar",
        ],
        default="all",
    )
    parser.add_argument("--export-dir", type=str, help="Directory for output files")
    parser.add_argument("--export-html", action="store_true", help="Export HTML dashboard")
    parser.add_argument(
        "--export-heatmap-csv",
        action="store_true",
        help="Export token-level heatmap CSV",
    )
    parser.add_argument(
        "--use-torch-overlay",
        action="store_true",
        help="Use Torch overlay instead of TF",
    )
    parser.add_argument("--launch-gui", action="store_true", help="Launch GUI interface")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")

    if args.launch_gui:
        launch_gui()
        return

    if args.file:
        file_path = Path(args.file)
        if args.annotate_file_before:
            try:
                logger.info(f"🧠 Annotating: {file_path}")
                original = file_path.read_text(encoding="utf-8")
                annotated = annotate_code_with_together_ai(original)
                file_path.write_text(annotated, encoding="utf-8")
                logger.info("✅ Annotation complete.")
            except Exception as e:
                logger.error(f"❌ Annotation failed: {e}")
                sys.exit(1)

        exit_code = analyse_file(
            file_path=file_path,
            raw=args.raw,
            format=args.format,
            out_path=Path(args.export_dir) if args.export_dir else None,
            json_out=Path(args.json_out) if args.json_out else None,
            csv_out=Path(args.csv_out) if args.csv_out else None,
            fail_threshold=args.fail_threshold,
            fail_on=args.fail_on,
            show_summary=args.summary or args.metrics_summary_table,
            save_summary_txt=(Path(args.save_summary_txt) if args.save_summary_txt else None),
            markdown_summary=args.metrics_summary_table,
            export_dir=Path(args.export_dir) if args.export_dir else None,
            export_html=args.export_html,
            export_heatmap_csv=args.export_heatmap_csv,
            use_torch_overlay=args.use_torch_overlay,
        )
        sys.exit(exit_code)

    elif args.dir:
        base_dir = Path(args.dir)
        failed = 0
        for file in base_dir.rglob("*.py"):
            logger.info(f"\n📄 {file}")
            result = analyse_file(file, raw=args.raw, format=args.format)
            if result != 0:
                failed += 1
        sys.exit(failed)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
