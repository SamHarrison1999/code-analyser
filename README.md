# Code Analyser — Rules ✕ RoBERTa with Gated Fusion (FastAPI)

*A calibrated hybrid of classic static analysis and an ML classifier for Python fintech code. Exposes a FastAPI microservice with `/healthz`, `/version`, `/predict`, and `/batch` for CI and local use.*

> **TL;DR**: Run rules **and** a multi‑label RoBERTa model; fuse them with simple, auditable gates to boost precision while keeping recall. Ship an API that works offline in CI and returns evidence for each finding.

---

## Features

- **Hybrid analysis**: rules (e.g., AST heuristics, linters) + **RoBERTa** multi‑label classifier
- **Gated fusion**: per‑label thresholds decide *confirm / add / veto* against rule hits
- **Calibration**: temperature scaling + PR‑sweeped thresholds stored in config
- **FastAPI microservice**: `/healthz`, `/version`, `/predict`, `/batch`
- **Deterministic & offline**: cache models and wheels; no outbound network in CI
- **Repro endpoints**: `/version` surfaces Python/OS, key package versions, git SHA, and model checksum

---

## Getting started

### 1) Requirements

- Python 3.10+ (3.12 tested)
- Windows, macOS, or Linux
- (Optional) Git, make

### 2) Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate              # macOS/Linux/WSL
# . .\venv\Scripts\activate         # Windows PowerShell (if you used "venv" not ".venv")
python -V && pip -V
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
# Recommended for locked installs:
# pip install --require-hashes -r requirements.txt
```

> **Offline tip**: prebuild wheels (`pip wheel -r requirements.txt -w wheels/`) and install with `--no-index --find-links wheels/`.

### 4) Start the API

Adjust the module path to your app entrypoint (examples below).

```bash
# Example: src.api:app or app.main:app — change to match your repo
uvicorn your_module:app --host 127.0.0.1 --port 8000 --workers 1
```

### 5) Smoke test (Bash)

```bash
# 1) Health
curl -fsS http://127.0.0.1:8000/healthz | jq .

# 2) Version (key fields)
curl -fsS http://127.0.0.1:8000/version | jq '{python_version, platform, fastapi, transformers, torch, sonarlint, pylint, model_sha256, git_commit}'

# 3) Predict on a tiny Python snippet
CODE=$'import subprocess\nsubprocess.run("ls", shell=True)'
jq -n --arg code "$CODE" --arg filename "risky.py" '{code:$code, filename:$filename}' | curl -fsS -H "Content-Type: application/json" -d @- http://127.0.0.1:8000/predict | jq .

# 4) Batch over files/paths
jq -n --arg path "examples/risky.py" '{paths:[$path]}' | curl -fsS -H "Content-Type: application/json" -d @- http://127.0.0.1:8000/batch | jq .
```

### 5′) Smoke test (PowerShell)

```powershell
# Health
Invoke-RestMethod -Uri "http://127.0.0.1:8000/healthz" -Method GET | ConvertTo-Json -Depth 5

# Version (subset)
$ver = Invoke-RestMethod -Uri "http://127.0.0.1:8000/version" -Method GET
$ver | Select-Object python_version, platform, fastapi, transformers, torch, sonarlint, pylint, model_sha256, git_commit | Format-List

# Predict
$code = @"
import subprocess
subprocess.run('dir', shell=True)
"@
$body = @{ code = $code; filename = "risky.py" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method POST -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 8

# Batch
$batch = @{ paths = @("examples\risky.py") } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/batch" -Method POST -ContentType "application/json" -Body $batch | ConvertTo-Json -Depth 8
```

---

## How it works

### High‑level flow
```
Code files (.py) ──▶ Rules ─┐
Code text (snippets) ─▶ Model ├─▶ Fusion ─▶ API (/predict, /batch)
                             └─▶ Evidence (rule IDs, probabilities, thresholds)
```

### Fusion policy (pseudocode)

```python
def fuse(rule_hit: bool, p: float, theta: float, margin: float = 0.0, gamma: float = 0.05):
    # confirm: rule fired and model agrees (>=θ)
    if rule_hit and p >= theta:
        return "confirm"
    # add: model is confident even if rule missed
    if not rule_hit and p >= theta and margin >= 0.0:
        return "add"
    # veto: rule fired but model is confidently below θ−γ
    if rule_hit and p <= (theta - gamma):
        return "veto"
    # otherwise leave unchanged
    return "pass"
```

### Configuration

Per‑label thresholds are loaded from config (example values shown here; adjust to your calibration):

```yaml
# config/thresholds.yaml
sast_risk:    {theta: 0.60, gamma: 0.05}
ml_signal:    {theta: 0.35, gamma: 0.05}
best_practice: {theta: 0.45, gamma: 0.05}
temperature:  1.00  # optional per-label temperatures
```

> **Calibration**: run temperature scaling on a held‑out split, then sweep θ on PR curves to meet your precision target; freeze before test/CI use.

---

## API (sketch)

> Real responses include IDs, messages, probabilities, and thresholds; structure may differ slightly depending on your codebase. Adjust examples accordingly.

### `GET /healthz`
- **200** `{ "status": "ok" }`

### `GET /version`
Returns versions and artefact identifiers for reproducibility.

```json
{
  "python_version": "3.12.1",
  "platform": "Windows-11",
  "fastapi": "0.115.0",
  "transformers": "4.43.3",
  "torch": "2.3.1",
  "sonarlint": "X.Y.Z",
  "pylint": "3.2.5",
  "model_sha256": "…",
  "git_commit": "abcdef0"
}
```

### `POST /predict`

```jsonc
# Request
{ "code": "def f(x):\return eval(x)", "filename": "risky.py" }

# Response (example)
{
	"predictions": [
	[
		{
		"label": "sast_risk",
		"score": 0.9046434164047241
		},
		{
		"label": "ml_signal",
		"score": 0.305173397064209
		},
		{
		"label": "best_practice",
		"score": 0.1459379345178604
		}
	],
	"id2label" {
		"0": "sast_risk",
		"1": "ml_signal",
		"2": "best_practice"
	},
	"threshold": 0.5,
	"messages": [
	"The key risk in this code is the use of 'eval()', which can execute arbitrary code and lead to security vulnerabilities such as code injection. A safer alternative is to use 'ast.literal_eval()' if the input is expected to be a Python literal expression (e.g., strings, numbers, tuples, lists, dicts). This approah safely evaluates the input without executing arbitrary code.\n\n '''python\nimport ast\ndef f(x):\n return ast.literal_eval(x)\n'''"
	]
}
```

### `POST /batch`

```jsonc
# Request: paths or raw code blobs
{ "paths": ["examples/"] }

# Response: list per-file
[{ "path": "examples/risky.py", "findings": [ ... ] }]
```

---

## Reproducibility checklist

- **Lock** dependencies (hash‑pinned or Poetry/uv):
  ```bash
  pip install --require-hashes -r requirements.txt
  ```
- **Expose** `/version` in your logs.
- **Seed** model & data loaders; cache model locally.
- **Disable network** during CI runs for deterministic tests.

---

**Datasets / artefacts**  
- An example labelled dataset lives on Hugging Face: <https://huggingface.co/datasets/SamH1999/fintech-ai-annotations/tree/main>
- Point your model to a local HF checkpoint <https://github.com/SamHarrison1999/code-analyser/tree/master/checkpoints/hf_retrained/phase2_full_ft>.

---

## CI example (GitHub Actions)

```yaml
name: analyse
on: [push, pull_request]
jobs:
  analyse:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.12' }
      - name: Install (locked)
        run: |
          python -m venv .venv
          source .venv/bin/activate
          pip install --upgrade pip
          pip install --require-hashes -r requirements.txt
      - name: Start API
        run: |
          source .venv/bin/activate
          uvicorn your_module:app --host 127.0.0.1 --port 8000 &
          sleep 5
      - name: Smoke test
        run: |
          curl -fsS http://127.0.0.1:8000/healthz
          curl -fsS http://127.0.0.1:8000/version
          python scripts/smoke_asserts.py
```

---

## Project structure (abridged)

```
code-analyser/
├── src/                     # package code
│   ├── api/                 # FastAPI app
│   ├── rules/               # rule runners & 
├── scripts/                 # CLI helpers (eval, batch, etc.)
├── examples/                # demo snippets
├── tests/                   # unit tests
└── README.md
```

---

## Contributing

Issues and PRs welcome. Please include:
- a failing test (if it’s a bug), or
- a brief rationale and before/after behaviour (if it’s a feature).

---

## Citing

If this work supports your research or teaching, please cite the project/dissertation. Example BibTeX (edit with your final thesis details):

```bibtex
@misc{harrison2025codeanalyser,
  title        = {AI-Enhanced Static Code Analysis with Gated Fusion},
  author       = {Harrison, Sam},
  year         = {2025},
  howpublished = {\url{https://github.com/SamHarrison1999/code-analyser}},
  note         = {Rules + RoBERTa with FastAPI}
}
```

---

## Acknowledgements

- Open-source authors of FastAPI, Transformers, PyTorch, and the linters used.
- University supervisor for feedback and evaluation support.
- The fintech OSS projects whose code segments informed the dataset.
