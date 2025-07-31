import os
import json
import shutil
import hashlib
import requests
import stat
from pathlib import Path
from dotenv import load_dotenv
from subprocess import run

# Load environment variables
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}

# Output path (clone directly here)
DEST_DIR = Path("datasets/github_fintech").resolve()
MANIFEST_PATH = DEST_DIR / ".manifest.json"
DEST_DIR.mkdir(parents=True, exist_ok=True)


def file_hash(filepath: Path) -> str:
    h = hashlib.sha256()
    with filepath.open("rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def save_manifest(manifest: dict):
    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def fetch_top_fintech_repos(max_repos=5) -> list:
    url = "https://api.github.com/search/repositories"
    params = {
        "q": "topic:fintech language:python",
        "sort": "stars",
        "order": "desc",
        "per_page": max_repos,
    }
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code == 200:
        return [
            {
                "name": item["name"],
                "clone_url": item["clone_url"],
                "html_url": item["html_url"],
            }
            for item in response.json().get("items", [])
        ]
    else:
        print(f"❌ GitHub API error {response.status_code}: {response.text}")
        return []


def force_remove_readonly(func, path, _):
    """Handle readonly file permissions on Windows."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def clean_non_python_files(repo_path: Path):
    # 1. Remove .git folder forcefully
    git_folder = repo_path / ".git"
    if git_folder.exists():
        try:
            shutil.rmtree(git_folder, onerror=force_remove_readonly)
        except Exception as e:
            print(f"⚠️ Could not delete .git folder: {e}")

    # 2. Delete all non-.py files
    for path in repo_path.rglob("*"):
        if path.is_file() and path.suffix != ".py":
            try:
                path.unlink()
            except Exception as e:
                print(f"⚠️ Failed to delete {path}: {e}")

    # 3. Remove empty folders
    for folder in sorted(repo_path.rglob("*"), reverse=True):
        if folder.is_dir() and not any(folder.iterdir()):
            try:
                folder.rmdir()
            except Exception as e:
                print(f"⚠️ Failed to remove empty folder {folder}: {e}")


def main():
    manifest = {}
    if MANIFEST_PATH.exists():
        with MANIFEST_PATH.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

    repos = fetch_top_fintech_repos()
    for repo in repos:
        target_path = DEST_DIR / repo["name"]
        if target_path.exists():
            print(f"📁 Skipping existing repo folder: {target_path}")
            continue

        print(f"⬇️ Cloning {repo['name']} into datasets/github_fintech/...")
        result = run(["git", "clone", "--depth=1", repo["clone_url"], str(target_path)])
        if result.returncode != 0:
            print(f"❌ Failed to clone {repo['clone_url']}")
            continue

        clean_non_python_files(target_path)

        for py_file in target_path.rglob("*.py"):
            rel_path = py_file.relative_to(DEST_DIR)
            hash_val = file_hash(py_file)
            manifest[str(rel_path)] = {
                "source_repo": repo["html_url"],
                "sha256": hash_val,
            }
            print(f"✅ Included: {rel_path}")

    save_manifest(manifest)
    print(f"✅ Finished cloning and filtering {len(repos)} repos.")


if __name__ == "__main__":
    main()
