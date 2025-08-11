# --- file: tools/calibrate_thresholds.py ---
# Annotation: This script calls your running microservice to collect probabilities on a validation set, fits a simple per-label temperature (on logits) and chooses F1-optimal thresholds; it saves both to JSON.
import argparse, json, csv, math, numpy as np, requests


# Annotation: Numerically safe logit transform with small epsilon to avoid infinities.
def prob_to_logit(p: float, eps: float = 1e-6) -> float:
    # Annotation: Clamp p to (eps, 1-eps).
    p = min(max(p, eps), 1.0 - eps)
    # Annotation: Return log(p/(1-p)).
    return math.log(p / (1.0 - p))


# Annotation: Sigmoid function to map a logit back to probability.
def sigmoid(z: float) -> float:
    # Annotation: Standard logistic function.
    return 1.0 / (1.0 + math.exp(-z))


# Annotation: Compute precision, recall and F1 at a given threshold.
def prf1(y: list[int], s: list[float], th: float) -> tuple[float, float, float]:
    # Annotation: True positives are positives with score ≥ threshold.
    tp = sum(int(yi and si >= th) for yi, si in zip(y, s))
    # Annotation: False positives are negatives with score ≥ threshold.
    fp = sum(int((not yi) and si >= th) for yi, si in zip(y, s))
    # Annotation: False negatives are positives with score < threshold.
    fn = sum(int(yi and si < th) for yi, si in zip(y, s))
    # Annotation: Precision with zero-division guard.
    P = tp / (tp + fp) if (tp + fp) else 0.0
    # Annotation: Recall with zero-division guard.
    R = tp / (tp + fn) if (tp + fn) else 0.0
    # Annotation: F1 with zero-division guard.
    F1 = 2 * P * R / (P + R) if (P + R) else 0.0
    # Annotation: Return the triple.
    return P, R, F1


# Annotation: Fit per-label temperature by grid search on validation logits to minimise BCE (good enough and stable).
def fit_temperature(y: list[int], p: list[float]) -> float:
    # Annotation: Convert probabilities to logits once.
    z = [prob_to_logit(pi) for pi in p]
    # Annotation: Search over a small grid of temperatures.
    grid = np.linspace(0.5, 3.0, 26)

    # Annotation: Define binary cross-entropy as a function of temperature T.
    def bce(T: float) -> float:
        s = [sigmoid(zi / T) for zi in z]
        eps = 1e-6
        return -sum(
            yi * math.log(si + eps) + (1 - yi) * math.log(1 - si + eps) for yi, si in zip(y, s)
        ) / len(y)

    # Annotation: Pick the temperature with the lowest BCE.
    best_T = min(grid, key=bce)
    # Annotation: Return the scalar temperature.
    return float(best_T)


# Annotation: Main routine that fetches scores from the service, calibrates T per label, then finds best thresholds and writes JSON.
def main():
    # Annotation: Define CLI flags.
    ap = argparse.ArgumentParser()
    # Annotation: Path to a validation CSV with 'text' and 'labels'.
    ap.add_argument("--val_csv", type=str, default="datasets/val/val.csv")
    # Annotation: Base URL for your model microservice.
    ap.add_argument("--url", type=str, default="http://127.0.0.1:8111/predict")
    # Annotation: Output folder to drop JSON files into.
    ap.add_argument("--out_dir", type=str, default="models/trained_model")
    # Annotation: Parse args.
    args = ap.parse_args()

    # Annotation: Fixed canonical label order.
    labels = ["sast_risk", "ml_signal", "best_practice"]
    # Annotation: Load validation rows.
    rows = list(csv.DictReader(open(args.val_csv, encoding="utf-8")))
    # Annotation: Extract raw texts.
    texts = [r["text"] for r in rows]
    # Annotation: Convert ground truth to per-label 0/1 vectors.
    truth = [set((r.get("labels") or "").split(";")) - {""} for r in rows]

    # Annotation: Pull model probabilities in small batches from the service with threshold=0.0 so we get raw scores.
    scores: list[dict] = []
    for i in range(0, len(texts), 32):
        payload = {"texts": texts[i : i + 32], "threshold": 0.0}
        res = requests.post(args.url, json=payload, timeout=30).json()
        scores.extend([x["scores"] for x in res["results"]])

    # Annotation: Collect per-label targets for calibration and thresholds.
    temps: dict[str, float] = {}
    thres: dict[str, float] = {}

    # Annotation: For each label, fit temperature and pick the F1-optimal threshold after temperature scaling.
    for lab in labels:
        y = [1 if (lab in t) else 0 for t in truth]
        p = [float(s.get(lab, 0.0)) for s in scores]
        T = fit_temperature(y, p)
        temps[lab] = round(T, 4)
        # Annotation: Apply temperature to logits and map back to probabilities.
        z = [prob_to_logit(pi) for pi in p]
        p_cal = [sigmoid(zi / T) for zi in z]
        # Annotation: Search a threshold grid to maximise F1.
        grid = np.linspace(0.05, 0.80, 76)
        best = max(grid, key=lambda th: prf1(y, p_cal, th)[2])
        thres[lab] = round(float(best), 4)

    # Annotation: Ensure output folder exists.
    os.makedirs(args.out_dir, exist_ok=True)
    # Annotation: Write temperature JSON.
    open(os.path.join(args.out_dir, "calibration.json"), "w", encoding="utf-8").write(
        json.dumps({"temperature": temps}, indent=2)
    )
    # Annotation: Write threshold JSON.
    open(os.path.join(args.out_dir, "thresholds.json"), "w", encoding="utf-8").write(
        json.dumps(thres, indent=2)
    )
    # Annotation: Print a small summary so you can paste into notes.
    print("Saved:", os.path.join(args.out_dir, "calibration.json"))
    print("Saved:", os.path.join(args.out_dir, "thresholds.json"))


# Annotation: Standard script guard.
if __name__ == "__main__":
    # Annotation: Run main when executed directly.
    main()
