# src/ml/huggingface_uploader.py

from huggingface_hub import HfApi, HfFolder, create_repo, upload_file
from pathlib import Path


def upload_zip_to_huggingface(zip_path: Path, repo_name: str, token: str = None):
    api = HfApi()
    token = token or HfFolder.get_token()

    if not token:
        raise ValueError(
            "🤖 HuggingFace token is required. Use `huggingface-cli login` or pass it explicitly."
        )

    if not zip_path.exists():
        raise FileNotFoundError(f"❌ ZIP file not found: {zip_path}")

    repo_id = f"{api.whoami(token)['name']}/{repo_name}"
    create_repo(repo_id, token=token, repo_type="dataset", exist_ok=True)

    upload_file(
        path_or_fileobj=zip_path,
        path_in_repo=zip_path.name,
        repo_id=repo_id,
        token=token,
        repo_type="dataset",
    )

    print(
        f"✅ Uploaded to HuggingFace: https://huggingface.co/datasets/{repo_id}/blob/main/{zip_path.name}"
    )
    return f"https://huggingface.co/datasets/{repo_id}/blob/main/{zip_path.name}"
