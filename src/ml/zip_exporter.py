# src/ml/zip_exporter.py

import zipfile
from pathlib import Path


def create_export_zip(source_dir: Path, zip_path: Path = None):
    if zip_path is None:
        zip_path = source_dir / "annotated_fintech_bundle.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in source_dir.rglob("*"):
            if file_path.is_file() and not file_path.name.endswith(".zip"):
                arcname = file_path.relative_to(source_dir)
                zipf.write(file_path, arcname)

    return zip_path
