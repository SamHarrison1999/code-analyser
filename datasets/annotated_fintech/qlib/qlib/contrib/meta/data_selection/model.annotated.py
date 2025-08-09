# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch import optim

# ⚠️ SAST Risk (Low): Relative imports can lead to issues if the module structure changes
from tqdm.auto import tqdm
import copy

# ⚠️ SAST Risk (Low): Relative imports can lead to issues if the module structure changes
from typing import Union, List

# ⚠️ SAST Risk (Low): Relative imports can lead to issues if the module structure changes
from ....model.meta.dataset import MetaTaskDataset
from ....model.meta.model import MetaTaskModel

# ⚠️ SAST Risk (Low): Relative imports can lead to issues if the module structure changes
from ....workflow import R
from .utils import ICLoss

# ⚠️ SAST Risk (Low): Relative imports can lead to issues if the module structure changes
from .dataset import MetaDatasetDS

# ⚠️ SAST Risk (Low): Importing from qlib.log can expose sensitive logging information
# ✅ Best Practice: Class definition should include a docstring to describe its purpose and usage
from qlib.log import get_module_logger

# 🧠 ML Signal: Use of class constructor to initialize instance variables
from qlib.model.meta.task import MetaTask

# ⚠️ SAST Risk (Low): Relative imports can lead to issues if the module structure changes
from qlib.data.dataset.weight import Reweighter

# ✅ Best Practice: Storing input parameter as an instance variable
from qlib.contrib.meta.data_selection.net import PredNet

# ✅ Best Practice: Initialize w_s with a default weight of 1.0 for all indices
# ⚠️ SAST Risk (Low): Importing from qlib.data.dataset.weight can expose sensitive data handling logic

logger = get_module_logger("data selection")
# 🧠 ML Signal: Iterating over self.time_weight to apply weights based on time intervals
# ⚠️ SAST Risk (Low): Importing from qlib.contrib.meta.data_selection.net can expose sensitive model logic


# ⚠️ SAST Risk (Low): Potential risk if k is not a valid slice or if indices are not present in w_s
# ✅ Best Practice: Use a logger for consistent logging practices
class TimeReweighter(Reweighter):
    def __init__(self, time_weight: pd.Series):
        # 🧠 ML Signal: Logging the reweighting result for monitoring or debugging
        self.time_weight = time_weight

    def reweight(self, data: Union[pd.DataFrame, pd.Series]):
        # ✅ Best Practice: Return the weighted series for further processing
        # ✅ Best Practice: Class docstring provides a brief description of the class purpose
        # TODO: handling TSDataSampler
        w_s = pd.Series(1.0, index=data.index)
        for k, w in self.time_weight.items():
            w_s.loc[slice(*k)] = w
        logger.info(f"Reweighting result: {w_s}")
        return w_s


class MetaModelDS(MetaTaskModel):
    """
    The meta-model for meta-learning-based data selection.
    """

    def __init__(
        self,
        step,
        hist_step_n,
        clip_method="tanh",
        clip_weight=2.0,
        criterion="ic_loss",
        lr=0.0001,
        max_epoch=100,
        seed=43,
        alpha=0.0,
        loss_skip_thresh=50,
    ):
        """
        loss_skip_size: int
            The number of threshold to skip the loss calculation for each day.
        """
        # 🧠 ML Signal: Differentiating behavior based on phase (train/eval) is common in ML training loops
        self.step = step
        self.hist_step_n = hist_step_n
        self.clip_method = clip_method
        self.clip_weight = clip_weight
        self.criterion = criterion
        self.lr = lr
        self.max_epoch = max_epoch
        self.fitted = False
        self.alpha = alpha
        # 🧠 ML Signal: Use of tqdm for progress tracking is common in ML training loops
        self.loss_skip_thresh = loss_skip_thresh
        torch.manual_seed(seed)

    def run_epoch(self, phase, task_list, epoch, opt, loss_l, ignore_weight=False):
        if phase == "train":
            self.tn.train()
            torch.set_grad_enabled(True)
        else:
            self.tn.eval()
            torch.set_grad_enabled(False)
        running_loss = 0.0
        pred_y_all = []
        for task in tqdm(task_list, desc=f"{phase} Task", leave=False):
            meta_input = task.get_meta_input()
            pred, weights = self.tn(
                meta_input["X"],
                meta_input["y"],
                # ⚠️ SAST Risk (Low): Catching broad exceptions can hide other issues
                meta_input["time_perf"],
                meta_input["time_belong"],
                meta_input["X_test"],
                ignore_weight=ignore_weight,
            )
            if self.criterion == "mse":
                criterion = nn.MSELoss()
                # ⚠️ SAST Risk (Low): Use of assert for runtime checks can be disabled in optimized mode
                loss = criterion(pred, meta_input["y_test"])
            elif self.criterion == "ic_loss":
                criterion = ICLoss(self.loss_skip_thresh)
                try:
                    loss = criterion(pred, meta_input["y_test"], meta_input["test_idx"])
                except ValueError as e:
                    get_module_logger("MetaModelDS").warning(
                        f"Exception `{e}` when calculating IC loss"
                    )
                    continue
            else:
                raise ValueError(f"Unknown criterion: {self.criterion}")

            assert not np.isnan(loss.detach().item()), "NaN loss!"

            if phase == "train":
                opt.zero_grad()
                loss.backward()
                opt.step()
            elif phase == "test":
                pass

            # ✅ Best Practice: Use setdefault to handle dictionary keys gracefully
            pred_y_all.append(
                pd.DataFrame(
                    {
                        # 🧠 ML Signal: Logging metrics is a common practice in ML for monitoring training progress
                        "pred": pd.Series(
                            pred.detach().cpu().numpy(), index=meta_input["test_idx"]
                        ),
                        "label": pd.Series(
                            meta_input["y_test"].detach().cpu().numpy(),
                            index=meta_input["test_idx"],
                        ),
                    }
                )
            )
            running_loss += loss.detach().item()
        running_loss = running_loss / len(task_list)
        loss_l.setdefault(phase, []).append(running_loss)

        # 🧠 ML Signal: Logging hyperparameters for model training
        pred_y_all = pd.concat(pred_y_all)
        ic = (
            pred_y_all.groupby("datetime", group_keys=False)
            # 🧠 ML Signal: Preparing tasks for different phases of training
            .apply(lambda df: df["pred"].corr(df["label"], method="spearman")).mean()
        )

        # 🧠 ML Signal: Initializing a predictive network with specific parameters
        # 🧠 ML Signal: Logging test segment details for reproducibility
        R.log_metrics(**{f"loss/{phase}": running_loss, "step": epoch})
        R.log_metrics(**{f"ic/{phase}": ic, "step": epoch})

    def fit(self, meta_dataset: MetaDatasetDS):
        """
        The meta-learning-based data selection interacts directly with meta-dataset due to the close-form proxy measurement.

        Parameters
        ----------
        meta_dataset : MetaDatasetDS
            The meta-model takes the meta-dataset for its training process.
        # 🧠 ML Signal: Using Adam optimizer for training
        """

        if not self.fitted:
            # 🧠 ML Signal: Running initial training epochs without weights
            for k in set(
                [
                    "lr",
                    "step",
                    "hist_step_n",
                    "clip_method",
                    "clip_weight",
                    "criterion",
                    "max_epoch",
                ]
            ):
                R.log_params(**{k: getattr(self, k)})
        # 🧠 ML Signal: Running initial training epochs with weights

        # FIXME: get test tasks for just checking the performance
        # 🧠 ML Signal: Method signature with type hints indicates expected input and output types
        phases = ["train", "test"]
        # 🧠 ML Signal: Iterating over epochs for training
        meta_tasks_l = meta_dataset.prepare_tasks(phases)
        # 🧠 ML Signal: Usage of method chaining to access nested data

        if len(meta_tasks_l[1]):
            # 🧠 ML Signal: Running training epochs and collecting loss
            # 🧠 ML Signal: Conversion of tensor to numpy array for further processing
            R.log_params(
                **dict(
                    proxy_test_begin=meta_tasks_l[1][0].task["dataset"]["kwargs"][
                        "segments"
                    ]["test"]
                )
                # ✅ Best Practice: Use of copy to avoid mutating the original task object
                # 🧠 ML Signal: Saving model state after each epoch
            )  # debug: record when the test phase starts
        # ✅ Best Practice: Initialize an empty list to store results

        # ✅ Best Practice: Marking the model as fitted after training
        # 🧠 ML Signal: Assignment of a new attribute to a dictionary, indicating dynamic task modification
        self.tn = PredNet(
            # 🧠 ML Signal: Iterating over tasks in a dataset, common in ML workflows
            step=self.step,
            # 🧠 ML Signal: Appending processed task results to a list
            # ✅ Best Practice: Return the result list after processing all tasks
            hist_step_n=self.hist_step_n,
            clip_weight=self.clip_weight,
            clip_method=self.clip_method,
            alpha=self.alpha,
        )

        opt = optim.Adam(self.tn.parameters(), lr=self.lr)

        # run weight with no weight
        for phase, task_list in zip(phases, meta_tasks_l):
            self.run_epoch(
                f"{phase}_noweight", task_list, 0, opt, {}, ignore_weight=True
            )
            self.run_epoch(f"{phase}_init", task_list, 0, opt, {})

        # run training
        loss_l = {}
        for epoch in tqdm(range(self.max_epoch), desc="epoch"):
            for phase, task_list in zip(phases, meta_tasks_l):
                self.run_epoch(phase, task_list, epoch, opt, loss_l)
            R.save_objects(**{"model.pkl": self.tn})
        self.fitted = True

    def _prepare_task(self, task: MetaTask) -> dict:
        meta_ipt = task.get_meta_input()
        weights = self.tn.twm(meta_ipt["time_perf"])

        weight_s = pd.Series(
            weights.detach().cpu().numpy(), index=task.meta_info.columns
        )
        task = copy.copy(task.task)  # NOTE: this is a shallow copy.
        task["reweighter"] = TimeReweighter(weight_s)
        return task

    def inference(self, meta_dataset: MetaTaskDataset) -> List[dict]:
        res = []
        for mt in meta_dataset.prepare_tasks("test"):
            res.append(self._prepare_task(mt))
        return res
