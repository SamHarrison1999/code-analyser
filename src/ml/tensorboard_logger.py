# File: code_analyser/src/ml/tensorboard_logger.py

import os
import logging
from typing import List, Dict
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

logger = logging.getLogger(__name__)


def log_metric_bundle_to_tensorboard(
    bundle: List[Dict[str, object]],
    run_name: str = "ast_metrics",
    step: int = 0,
    log_dir: str = "runs",
):
    """
    Log a list of AST metrics (with confidence + severity) to TensorBoard.

    Args:
        bundle (List[Dict]): [{'metric': str, 'value': int, 'confidence': float, 'severity': str}, ...]
        run_name (str): TensorBoard subdirectory name
        step (int): Current training or epoch step
        log_dir (str): Root logging directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=os.path.join(log_dir, run_name, timestamp))

    for entry in bundle:
        name = entry.get("metric", "")
        val = float(entry.get("value", 0))
        conf = float(entry.get("confidence", 0))
        severity = entry.get("severity", "").lower()

        # Scalar metrics
        writer.add_scalar(f"{name}/value", val, step)
        writer.add_scalar(f"{name}/confidence", conf, step)

        # One-hot severity
        writer.add_scalar(f"{name}/severity_low", 1.0 if severity == "low" else 0.0, step)
        writer.add_scalar(f"{name}/severity_medium", 1.0 if severity == "medium" else 0.0, step)
        writer.add_scalar(f"{name}/severity_high", 1.0 if severity == "high" else 0.0, step)

    writer.flush()
    writer.close()
