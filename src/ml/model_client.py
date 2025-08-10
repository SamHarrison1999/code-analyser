# --- file: src/ml/model_client.py ---
# Annotation: This lightweight client wraps HTTP calls to the FastAPI model service with minimal dependencies.
from typing import List, Dict, Any, Optional

# Annotation: Requests provides a reliable HTTP client; it's the only runtime dependency here.
import requests


# Annotation: We avoid importing from 'ml' or other project modules to prevent circular imports and path errors.
# Annotation: AnalyserModelClient exposes a thin predict() wrapper around POST /predict.
class AnalyserModelClient:
    # Annotation: base_url is the service root (e.g., http://127.0.0.1:8111); timeout controls request timeouts in seconds; session enables connection reuse.
    def __init__(
        self, base_url: str, timeout: float = 60.0, session: Optional[requests.Session] = None
    ):
        # Annotation: Normalise and store the base URL without a trailing slash to avoid '//' when joining.
        self.base_url = base_url.rstrip("/")
        # Annotation: Store the timeout to be applied to all requests.
        self.timeout = timeout
        # Annotation: Use a shared Session for efficient connection pooling across multiple calls.
        self.session = session or requests.Session()

    # Annotation: predict sends a batch of texts and returns the parsed JSON response from the service.
    def predict(self, texts: List[str], threshold: float = 0.5) -> Dict[str, Any]:
        # Annotation: Construct the JSON payload expected by the /predict endpoint.
        payload = {"texts": texts, "threshold": threshold}
        # Annotation: Issue the POST request and check for HTTP errors.
        resp = self.session.post(f"{self.base_url}/predict", json=payload, timeout=self.timeout)
        # Annotation: Raise an exception for non-2xx responses so callers can handle failures explicitly.
        resp.raise_for_status()
        # Annotation: Return the parsed JSON body as a Python dictionary.
        return resp.json()

    # Annotation: health performs a simple GET /healthz to check service availability and metadata.
    def health(self) -> Dict[str, Any]:
        # Annotation: Issue the GET request to the health endpoint with a shorter timeout for responsiveness.
        resp = self.session.get(f"{self.base_url}/healthz", timeout=min(self.timeout, 10.0))
        # Annotation: Convert HTTP errors into exceptions for clearer failure modes.
        resp.raise_for_status()
        # Annotation: Return the parsed health payload.
        return resp.json()


if __name__ == "__main__":
    # Annotation: Basic CLI smoke test to verify connectivity when running `python -m src.ml.model_client`.
    import sys
    import json

    # Annotation: Default to localhost unless a URL is provided as the first CLI argument.
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8111"
    # Annotation: Instantiate the client and print health plus a sample prediction to stdout.
    client = AnalyserModelClient(base_url=url)
    print(json.dumps(client.health(), indent=2))
    print(json.dumps(client.predict(["print(1)"], threshold=0.5), indent=2))
