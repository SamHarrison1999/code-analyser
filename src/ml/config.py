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

MODEL_CONFIG = {
    "model_name": "microsoft/codebert-base",
    "vocab_size": 10000,
    "embed_dim": 128,
    "hidden_dim": 256,
    "output_dim": 3,  # SAST, ML Signal, Best Practice
    "dropout": 0.3,
    "use_attention": True,
    "use_hf": True,
    "use_distilled": False,
}

TRAINING_CONFIG = {
    "epochs": 3,
    "batch_size": 8,
    "learning_rate": 2e-5,
    "max_length": 512,
    "confidence_threshold": 0.7,
    "max_train_samples": None,  # Set to an int to limit training samples
    "log_dir": str(LOG_DIR.as_posix()),
    "output_dir": str(CHECKPOINT_DIR.as_posix()),
    "seed": 42,
    "stratify": True,
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
}
