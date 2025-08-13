# --- file: src/ml/model_client.py ---
# This lightweight client wraps HTTP calls to the FastAPI model service with minimal dependencies.
from typing import List, Dict, Any, Optional
# Requests provides a reliable HTTP client; it's the only runtime dependency here.
import requests
# AnalyserModelClient exposes helpers around the model service endpoints.
class AnalyserModelClient:
    # base_url is the service root (e.g., http://127.0.0.1:8111); timeout controls request timeouts in seconds; session enables connection reuse.
    def __init__(self, base_url: str, timeout: float = 60.0, session: Optional[requests.Session] = None):
        # Normalise and store the base URL without a trailing slash to avoid '//' when joining.
        self.base_url = base_url.rstrip("/")
        # Store the timeout to be applied to all requests.
        self.timeout = timeout
        # Use a shared Session for efficient connection pooling across multiple calls.
        self.session = session or requests.Session()
    # predict sends a batch of texts and returns the parsed JSON response from the service. Set explain=True to request OpenAI messages per text.
    def predict(self, texts: List[str], threshold: float = 0.5, explain: bool = True) -> Dict[str, Any]:
        # Construct the JSON payload expected by the /predict endpoint.
        payload = {"texts": texts, "threshold": threshold, "explain": explain}
        # Issue the POST request and check for HTTP errors.
        resp = self.session.post(f"{self.base_url}/predict", json=payload, timeout=self.timeout)
        # Raise an exception for non-2xx responses so callers can handle failures explicitly.
        resp.raise_for_status()
        # Return the parsed JSON body as a Python dictionary (includes 'messages' when available).
        return resp.json()
    # health performs a simple GET /healthz to check service availability and metadata.
    def health(self) -> Dict[str, Any]:
        # Issue the GET request to the health endpoint with a shorter timeout for responsiveness.
        resp = self.session.get(f"{self.base_url}/healthz", timeout=min(self.timeout, 10.0))
        # Convert HTTP errors into exceptions for clearer failure modes.
        resp.raise_for_status()
        # Return the parsed health payload.
        return resp.json()
    # version fetches diagnostics from GET /version for environment and artefact provenance.
    def version(self) -> Dict[str, Any]:
        # Issue the GET request to the version endpoint with a shorter timeout for responsiveness.
        resp = self.session.get(f"{self.base_url}/version", timeout=min(self.timeout, 10.0))
        # Surface non-2xx responses to the caller for explicit handling.
        resp.raise_for_status()
        # Return parsed JSON with version and checksum metadata.
        return resp.json()
    # batch sends local file paths to the service for batched processing and returns the job summary with a download URL. Set explain=True to request OpenAI messages.
    def batch(self, paths: List[str], write_fixed: bool = True, explain: bool = True) -> Dict[str, Any]:
        # Construct the JSON payload expected by the /batch endpoint.
        payload = {"paths": paths, "write_fixed": write_fixed, "explain": explain}
        # Issue the POST request and check for HTTP errors.
        resp = self.session.post(f"{self.base_url}/batch", json=payload, timeout=self.timeout)
        # Raise if non-2xx so the caller can handle failures.
        resp.raise_for_status()
        # Return the parsed JSON response containing job_id, results, and archive_url.
        return resp.json()
    # download_batch downloads the batch archive to the given destination path and returns that path.
    def download_batch(self, job_id: str, dest_path: str) -> str:
        # Stream the archive to avoid loading large zips fully into memory.
        url = f"{self.base_url}/batch/{job_id}/download"
        with self.session.get(url, stream=True, timeout=self.timeout) as r:
            r.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        # Return the local file path for convenience.
        return dest_path
if __name__ == "__main__":
    # Basic CLI smoke test to verify connectivity when running `python -m src.ml.model_client`.
    import sys, json, os
    # Default to localhost unless a URL is provided as the first CLI argument.
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8111"
    # Instantiate the client and print health plus a sample prediction to stdout (includes messages if enabled).
    client = AnalyserModelClient(base_url=url)
    print(json.dumps(client.health(), indent=2))
    print(json.dumps(client.predict(["print(1)"], threshold=0.5, explain=True), indent=2))
    # Run a tiny batch over this file if it exists, then print the batch summary (including messages if OpenAI is enabled).
    sample_path = __file__ if os.path.exists(__file__) else "README.md"
    batch = client.batch([sample_path], explain=True)
    print(json.dumps(batch, indent=2))
    # Optionally download the archive (commented by default).
    # job_id = batch.get("job_id")
    # if job_id:
    #     out_zip = f"batch_{job_id}.zip"
    #     print("Saved:", client.download_batch(job_id, out_zip))