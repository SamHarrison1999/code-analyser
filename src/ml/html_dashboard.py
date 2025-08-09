# src/ml/html_dashboard.py

import json
from pathlib import Path


def write_html_dashboard(output_dir: Path, dashboard_path: Path):
    output_dir = Path(output_dir)
    annotation_files = list(output_dir.rglob("*_annotations.json"))
    if not annotation_files:
        raise ValueError(f"No annotation JSON files found in: {output_dir}")

    label_counts = {"sast_risk": 0, "ml_signal": 0, "best_practice": 0, "unknown": 0}
    severity_counts = {"High": 0, "Medium": 0, "Low": 0, "Unknown": 0}
    confidence_values = []
    file_summaries = []

    def is_valid_reason(reason: str) -> bool:
        if not reason:
            return False
        reason = reason.strip().lower()
        return reason and not (
            reason.startswith("none") or "n/a" in reason or "not applicable" in reason
        )

    for ann_path in annotation_files:
        with ann_path.open(encoding="utf-8") as f:
            annotations = json.load(f)

        valid_annotations = [a for a in annotations if is_valid_reason(a.get("reason", ""))]
        skipped = len(annotations) - len(valid_annotations)
        if skipped:
            print(f"⚠️ Skipped {skipped} invalid annotations in {ann_path.name}")

        total = len(valid_annotations)
        avg_conf = round(
            sum(a.get("confidence", 1.0) for a in valid_annotations) / total, 3
        ) if total else 0.0

        confidence_values.append(avg_conf)

        for ann in valid_annotations:
            label = ann.get("label", "unknown")
            if label not in label_counts:
                label_counts["unknown"] += 1
            else:
                label_counts[label] += 1

            if label == "sast_risk":
                sev = ann.get("severity", "Unknown")
                if sev not in severity_counts:
                    severity_counts["Unknown"] += 1
                else:
                    severity_counts[sev] += 1

        file_summaries.append({
            "file": ann_path.relative_to(output_dir).as_posix(),
            "total_annotations": total,
            "average_confidence": avg_conf,
        })

    total_files = len(file_summaries)
    avg_conf_overall = round(sum(confidence_values) / total_files, 3) if total_files else 0.0

    html = f"""
    <html>
    <head>
        <title>Annotation Dashboard</title>
        <style>
            body {{ font-family: sans-serif; margin: 2em; }}
            h1 {{ color: #444; }}
            .section {{ margin-top: 2em; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ccc; padding: 0.5em; text-align: left; }}
            th {{ background: #eee; }}
        </style>
    </head>
    <body>
        <h1>📊 Annotation Summary</h1>
        <div class="section">
            <h2>Total Files Annotated: {total_files}</h2>
            <h3>Average Confidence: {avg_conf_overall}</h3>
        </div>

        <div class="section">
            <h2>Annotation Types</h2>
            <ul>
                <li>⚠️ SAST Risk: {label_counts['sast_risk']}</li>
                <li>🧠 ML Signal: {label_counts['ml_signal']}</li>
                <li>✅ Best Practice: {label_counts['best_practice']}</li>
                <li>❓ Unknown: {label_counts['unknown']}</li>
            </ul>
        </div>

        <div class="section">
            <h2>Severity Breakdown (for SAST Risk)</h2>
            <ul>
                <li>High: {severity_counts['High']}</li>
                <li>Medium: {severity_counts['Medium']}</li>
                <li>Low: {severity_counts['Low']}</li>
                <li>Unknown: {severity_counts['Unknown']}</li>
            </ul>
        </div>

        <div class="section">
            <h2>File Summary</h2>
            <table>
                <tr><th>File</th><th># Annotations</th><th>Avg. Confidence</th></tr>
                {''.join(f"<tr><td>{a['file']}</td><td>{a['total_annotations']}</td><td>{a['average_confidence']:.2f}</td></tr>" for a in file_summaries)}
            </table>
        </div>
    </body>
    </html>
    """

    dashboard_path.write_text(html.strip(), encoding="utf-8")

