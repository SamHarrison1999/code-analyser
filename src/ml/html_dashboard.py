# src/ml/html_dashboard.py

from pathlib import Path
import json


def write_html_dashboard(output_dir: Path, html_path: Path):
    annotation_files = list(output_dir.rglob("*.annotations.json"))
    if not annotation_files:
        raise ValueError(f"No annotation JSON files found in: {output_dir}")

    annotation_types = {"⚠️ SAST Risk": 0, "🧠 ML Signal": 0, "✅ Best Practice": 0}
    severity_count = {"High": 0, "Medium": 0, "Low": 0}
    confidence_values = []
    file_summaries = []

    def is_valid_reason(reason: str) -> bool:
        if not reason:
            return False
        reason = reason.strip().lower()
        return not (
            reason.startswith("none") or "n/a" in reason or "not applicable" in reason
        )

    for ann_path in annotation_files:
        with ann_path.open(encoding="utf-8") as f:
            annotations = json.load(f)

        valid_annotations = [
            a for a in annotations if is_valid_reason(a.get("reason", ""))
        ]
        total = len(valid_annotations)
        avg_conf = 1.0 if total > 0 else 0.0

        confidence_values.append(avg_conf)

        for ann in valid_annotations:
            typ = ann.get("type")
            if typ in annotation_types:
                annotation_types[typ] += 1
            if typ == "⚠️ SAST Risk":
                sev = ann.get("severity")
                if sev in severity_count:
                    severity_count[sev] += 1

        file_summaries.append(
            {
                "file": ann_path.relative_to(output_dir).as_posix(),
                "total_annotations": total,
                "average_confidence": avg_conf,
            }
        )

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
        <h1>📊 Together.ai Annotation Summary</h1>
        <div class="section">
            <h2>Total Files Annotated: {len(file_summaries)}</h2>
            <h3>Average Confidence: {sum(confidence_values)/len(confidence_values):.3f}</h3>
        </div>

        <div class="section">
            <h2>Annotation Types</h2>
            <ul>
                <li>⚠️ SAST Risk: {annotation_types['⚠️ SAST Risk']}</li>
                <li>🧠 ML Signal: {annotation_types['🧠 ML Signal']}</li>
                <li>✅ Best Practice: {annotation_types['✅ Best Practice']}</li>
            </ul>
        </div>

        <div class="section">
            <h2>Severity Breakdown</h2>
            <ul>
                <li>High: {severity_count['High']}</li>
                <li>Medium: {severity_count['Medium']}</li>
                <li>Low: {severity_count['Low']}</li>
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

    html_path.write_text(html.strip(), encoding="utf-8")
