# ✅ Best Practice: Central configuration file for ML model, training, inference, and export parameters
# 🧠 ML Signal: These values influence learning dynamics, capacity, and annotation behaviour

from pathlib import Path

# 📁 Automatically resolve and ensure key directories exist
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = PROJECT_ROOT / "datasets" / "github_fintech"
ANNOTATION_DIR = PROJECT_ROOT / "datasets" / "annotated_fintech"
AI_CACHE_DIR = PROJECT_ROOT / ".ai_cache"
LOG_DIR = PROJECT_ROOT / "logs" / "tensorboard"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "local_finetuned"
ANNOTATION_OUTPUT_DIR = PROJECT_ROOT / "ai_annotations"

# ✅ Ensure all necessary folders exist
for directory in [
    DATASET_DIR,
    ANNOTATION_DIR,
    AI_CACHE_DIR,
    LOG_DIR,
    CHECKPOINT_DIR,
    ANNOTATION_OUTPUT_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

TRAINING_CONFIG = {
    "output_dir": "checkpoints/supervised",
    "use_tensorboard": True,
    "use_hf": True,
    "epochs": 10,
    "batch_size": 8,
    "learning_rate": 2e-5,
}
MODEL_CONFIG = {
    "model_name": "microsoft/codebert-base",
    "hidden_size": 768,
    "num_labels": 3,
}


INFERENCE_CONFIG = {
    "threshold": 0.5,  # used for multi-label classification
    "annotation_output_dir": str(ANNOTATION_OUTPUT_DIR.as_posix()),
    "use_heatmaps": True,
}

EXPORT_CONFIG = {
    "torchscript_path": str(
        (PROJECT_ROOT / "checkpoints" / "torchscript_model.pt").as_posix()
    ),
    "onnx_path": str((PROJECT_ROOT / "checkpoints" / "onnx_model.onnx").as_posix()),
}

DATA_PATHS = {
    "code_dir": str(DATASET_DIR.as_posix()),
    "annotation_dir": str(ANNOTATION_DIR.as_posix()),
    "processed_dataset": str((PROJECT_ROOT / "datasets" / "processed").as_posix()),
    "supervised_ckpt": "checkpoints/supervised",
    "label_map": {
        "sast_risk": 0,
        "ml_signal": 1,
        "best_practice": 2
    }
}

