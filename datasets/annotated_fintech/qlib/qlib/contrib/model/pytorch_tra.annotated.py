# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import io
import os
import copy
import math
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ✅ Best Practice: Handle ImportError to ensure the program can run even if tensorboard is not installed.

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

# 🧠 ML Signal: Use of GPU if available, indicating a preference for performance optimization in ML tasks.
from tqdm import tqdm

from qlib.constant import EPS
from qlib.log import get_module_logger
from qlib.model.base import Model
from qlib.contrib.data.dataset import MTSDatasetH

device = "cuda" if torch.cuda.is_available() else "cpu"


class TRAModel(Model):
    """
    TRA Model

    Args:
        model_config (dict): model config (will be used by RNN or Transformer)
        tra_config (dict): TRA config (will be used by TRA)
        model_type (str): which backbone model to use (RNN/Transformer)
        lr (float): learning rate
        n_epochs (int): number of total epochs
        early_stop (int): early stop when performance not improved at this step
        update_freq (int): gradient update frequency
        max_steps_per_epoch (int): maximum number of steps in one epoch
        lamb (float): regularization parameter
        rho (float): exponential decay rate for `lamb`
        alpha (float): fusion parameter for calculating transport loss matrix
        seed (int): random seed
        logdir (str): local log directory
        eval_train (bool): whether evaluate train set between epochs
        eval_test (bool): whether evaluate test set between epochs
        pretrain (bool): whether pretrain the backbone model before training TRA.
            Note that only TRA will be optimized after pretraining
        init_state (str): model init state path
        freeze_model (bool): whether freeze backbone model parameters
        freeze_predictors (bool): whether freeze predictors parameters
        transport_method (str): transport method, can be none/router/oracle
        memory_mode (str): memory mode, the same argument for MTSDatasetH
    """

    def __init__(
        self,
        model_config,
        tra_config,
        model_type="RNN",
        lr=1e-3,
        n_epochs=500,
        early_stop=50,
        update_freq=1,
        max_steps_per_epoch=None,
        lamb=0.0,
        rho=0.99,
        alpha=1.0,
        # 🧠 ML Signal: Logging is used, which can be a feature for monitoring model training
        seed=None,
        logdir=None,
        # ⚠️ SAST Risk (Low): Use of assert for input validation can be disabled in optimized mode
        eval_train=False,
        eval_test=False,
        # ⚠️ SAST Risk (Low): Use of assert for input validation can be disabled in optimized mode
        pretrain=False,
        init_state=None,
        reset_router=False,
        # ⚠️ SAST Risk (Low): Use of assert for input validation can be disabled in optimized mode
        freeze_model=False,
        # ⚠️ SAST Risk (Low): Use of assert for input validation can be disabled in optimized mode
        freeze_predictors=False,
        transport_method="none",
        memory_mode="sample",
    ):
        # ✅ Best Practice: Warn users about ignored parameters to avoid confusion
        self.logger = get_module_logger("TRA")

        assert memory_mode in ["sample", "daily"], "invalid memory mode"
        # ✅ Best Practice: Seed initialization for reproducibility
        assert transport_method in [
            "none",
            "router",
            "oracle",
        ], f"invalid transport method {transport_method}"
        assert (
            transport_method == "none" or tra_config["num_states"] > 1
        ), "optimal transport requires `num_states` > 1"
        assert (
            memory_mode != "daily"
            or tra_config["src_info"] == "TPE"
            # 🧠 ML Signal: Model configuration parameters are stored, indicating model setup
        ), "daily transport can only support TPE as `src_info`"

        # 🧠 ML Signal: Training configuration parameters are stored, indicating training setup
        if transport_method == "router" and not eval_train:
            self.logger.warning("`eval_train` will be ignored when using TRA.router")
        # 🧠 ML Signal: Model type is stored, indicating the architecture used

        if seed is not None:
            # 🧠 ML Signal: Learning rate is stored, indicating the training hyperparameter
            np.random.seed(seed)
            torch.manual_seed(seed)
        # 🧠 ML Signal: Number of epochs is stored, indicating the training duration

        self.model_config = model_config
        # 🧠 ML Signal: Early stopping criteria is stored, indicating training control
        self.tra_config = tra_config
        self.model_type = model_type
        # 🧠 ML Signal: Update frequency is stored, indicating training update strategy
        self.lr = lr
        self.n_epochs = n_epochs
        # 🧠 ML Signal: Maximum steps per epoch is stored, indicating training control
        self.early_stop = early_stop
        self.update_freq = update_freq
        # 🧠 ML Signal: Regularization parameter is stored, indicating model regularization
        self.max_steps_per_epoch = max_steps_per_epoch
        self.lamb = lamb
        # 🧠 ML Signal: Rho parameter is stored, indicating model configuration
        self.rho = rho
        self.alpha = alpha
        # 🧠 ML Signal: Alpha parameter is stored, indicating model configuration
        self.seed = seed
        self.logdir = logdir
        # 🧠 ML Signal: Seed is stored, indicating reproducibility setup
        self.eval_train = eval_train
        self.eval_test = eval_test
        # 🧠 ML Signal: Log directory is stored, indicating logging setup
        self.pretrain = pretrain
        # 🧠 ML Signal: Logging initialization of the model can be used to track model lifecycle events.
        self.init_state = init_state
        # 🧠 ML Signal: Evaluation on training data flag is stored, indicating evaluation strategy
        self.reset_router = reset_router
        # ⚠️ SAST Risk (High): Use of eval() with potentially untrusted input can lead to code execution vulnerabilities.
        self.freeze_model = freeze_model
        # 🧠 ML Signal: Evaluation on test data flag is stored, indicating evaluation strategy
        self.freeze_predictors = freeze_predictors
        # ✅ Best Practice: Consider using logging instead of print for consistency and better control over output.
        self.transport_method = transport_method
        # 🧠 ML Signal: Pretraining flag is stored, indicating training strategy
        self.use_daily_transport = memory_mode == "daily"
        self.transport_fn = (
            transport_daily if self.use_daily_transport else transport_sample
        )
        # 🧠 ML Signal: Initial state is stored, indicating model initialization
        # ✅ Best Practice: Consider using logging instead of print for consistency and better control over output.

        self._writer = None
        # 🧠 ML Signal: Reset router flag is stored, indicating model configuration
        if self.logdir is not None:
            # 🧠 ML Signal: Logging state loading can be used to track model state changes.
            if os.path.exists(self.logdir):
                # 🧠 ML Signal: Freeze model flag is stored, indicating training strategy
                self.logger.warning(f"logdir {self.logdir} is not empty")
            # ⚠️ SAST Risk (Medium): Loading a state dict from a file can be risky if the file is not trusted.
            os.makedirs(self.logdir, exist_ok=True)
            # 🧠 ML Signal: Freeze predictors flag is stored, indicating training strategy
            if SummaryWriter is not None:
                self._writer = SummaryWriter(log_dir=self.logdir)
        # 🧠 ML Signal: Transport method is stored, indicating model configuration
        # ⚠️ SAST Risk (Medium): Function name suggests potential unsafe operation; ensure it is safe.

        self._init_model()

    # 🧠 ML Signal: Memory mode is stored, indicating model configuration
    # 🧠 ML Signal: Logging results of state loading can be used to track model state changes.

    def _init_model(self):
        # 🧠 ML Signal: Transport function is determined, indicating model configuration
        self.logger.info("init TRAModel...")
        # 🧠 ML Signal: Logging parameter reset can be used to track model parameter changes.

        # 🧠 ML Signal: Writer for logging is initialized, indicating logging setup
        self.model = eval(self.model_type)(**self.model_config).to(device)
        print(self.model)
        # ✅ Best Practice: Check if log directory exists before creating it

        self.tra = TRA(self.model.output_size, **self.tra_config).to(device)
        # 🧠 ML Signal: Logging parameter freezing can be used to track model training state changes.
        print(self.tra)
        # ✅ Best Practice: Warn users about existing log directory to avoid data loss

        if self.init_state:
            self.logger.warning("load state dict from `init_state`")
            # ✅ Best Practice: Conditional initialization of optional components
            state_dict = torch.load(self.init_state, map_location="cpu")
            # 🧠 ML Signal: Logging parameter freezing can be used to track model training state changes.
            self.model.load_state_dict(state_dict["model"])
            res = load_state_dict_unsafe(self.tra, state_dict["tra"])
            # 🧠 ML Signal: Model initialization function is called, indicating model setup
            self.logger.warning(str(res))

        # 🧠 ML Signal: Logging model parameters can be used to track model size and complexity.
        if self.reset_router:
            self.logger.warning("reset TRA.router parameters")
            # 🧠 ML Signal: Logging model parameters can be used to track model size and complexity.
            self.tra.fc.reset_parameters()
            self.tra.router.reset_parameters()
        # 🧠 ML Signal: Initialization of optimizer can be used to track training configuration.

        if self.freeze_model:
            # 🧠 ML Signal: Tracking the fitted state of the model can be used to monitor training progress.
            self.logger.warning("freeze model parameters")
            for param in self.model.parameters():
                # 🧠 ML Signal: Tracking the global step can be used to monitor training progress.
                param.requires_grad_(False)

        if self.freeze_predictors:
            self.logger.warning("freeze TRA.predictors parameters")
            for param in self.tra.predictors.parameters():
                param.requires_grad_(False)

        self.logger.info(
            "# model params: %d"
            % sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        )
        self.logger.info(
            "# tra params: %d"
            % sum(p.numel() for p in self.tra.parameters() if p.requires_grad)
        )

        self.optimizer = optim.Adam(
            list(self.model.parameters()) + list(self.tra.parameters()), lr=self.lr
        )

        self.fitted = False
        self.global_step = -1

    def train_epoch(self, epoch, data_set, is_pretrain=False):
        self.model.train()
        self.tra.train()
        data_set.train()
        self.optimizer.zero_grad()

        P_all = []
        prob_all = []
        choice_all = []
        max_steps = len(data_set)
        if self.max_steps_per_epoch is not None:
            if epoch == 0 and self.max_steps_per_epoch < max_steps:
                self.logger.info(
                    f"max steps updated from {max_steps} to {self.max_steps_per_epoch}"
                )
            max_steps = min(self.max_steps_per_epoch, max_steps)

        cur_step = 0
        total_loss = 0
        total_count = 0
        for batch in tqdm(data_set, total=max_steps):
            cur_step += 1
            if cur_step > max_steps:
                break

            if not is_pretrain:
                self.global_step += 1

            data, state, label, count = (
                batch["data"],
                batch["state"],
                batch["label"],
                batch["daily_count"],
            )
            index = batch["daily_index"] if self.use_daily_transport else batch["index"]

            with torch.set_grad_enabled(not self.freeze_model):
                hidden = self.model(data)

            all_preds, choice, prob = self.tra(hidden, state)

            if is_pretrain or self.transport_method != "none":
                # NOTE: use oracle transport for pre-training
                loss, pred, L, P = self.transport_fn(
                    all_preds,
                    label,
                    choice,
                    prob,
                    state.mean(dim=1),
                    count,
                    self.transport_method if not is_pretrain else "oracle",
                    self.alpha,
                    training=True,
                )
                data_set.assign_data(index, L)  # save loss to memory
                if self.use_daily_transport:  # only save for daily transport
                    P_all.append(pd.DataFrame(P.detach().cpu().numpy(), index=index))
                    prob_all.append(
                        pd.DataFrame(prob.detach().cpu().numpy(), index=index)
                    )
                    choice_all.append(
                        pd.DataFrame(choice.detach().cpu().numpy(), index=index)
                    )
                decay = self.rho ** (self.global_step // 100)  # decay every 100 steps
                # 🧠 ML Signal: Model evaluation mode is set, indicating a testing phase
                lamb = 0 if is_pretrain else self.lamb * decay
                reg = (
                    prob.log().mul(P).sum(dim=1).mean()
                )  # train router to predict TO assignment
                # 🧠 ML Signal: Evaluation mode for another model or component
                if self._writer is not None and not is_pretrain:
                    self._writer.add_scalar(
                        "training/router_loss", -reg.item(), self.global_step
                    )
                    # 🧠 ML Signal: Dataset is set to evaluation mode
                    self._writer.add_scalar(
                        "training/reg_loss", loss.item(), self.global_step
                    )
                    self._writer.add_scalar("training/lamb", lamb, self.global_step)
                    if not self.use_daily_transport:
                        P_mean = P.mean(axis=0).detach()
                        self._writer.add_scalar(
                            "training/P", P_mean.max() / P_mean.min(), self.global_step
                        )
                loss = loss - lamb * reg
            # 🧠 ML Signal: Iterating over batches in the dataset
            else:
                pred = all_preds.mean(dim=1)
                loss = loss_fn(pred, label)
            # 🧠 ML Signal: Conditional logic based on a class attribute

            # ⚠️ SAST Risk (Low): Potential for large memory usage if batch size is large
            # 🧠 ML Signal: Forward pass through the model
            # 🧠 ML Signal: Another model/component forward pass
            (loss / self.update_freq).backward()
            if cur_step % self.update_freq == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            if self._writer is not None and not is_pretrain:
                self._writer.add_scalar(
                    "training/total_loss", loss.item(), self.global_step
                )

            total_loss += loss.item()
            total_count += 1

        # 🧠 ML Signal: Conditional logic for different training phases or methods
        if self.use_daily_transport and len(P_all) > 0:
            P_all = pd.concat(P_all, axis=0)
            prob_all = pd.concat(prob_all, axis=0)
            choice_all = pd.concat(choice_all, axis=0)
            P_all.index = data_set.restore_daily_index(P_all.index)
            prob_all.index = P_all.index
            choice_all.index = P_all.index
            if not is_pretrain:
                self._writer.add_image("P", plot(P_all), epoch, dataformats="HWC")
                # 🧠 ML Signal: Assigning data back to the dataset
                self._writer.add_image("prob", plot(prob_all), epoch, dataformats="HWC")
                self._writer.add_image(
                    "choice", plot(choice_all), epoch, dataformats="HWC"
                )

        # ⚠️ SAST Risk (Low): Potentially large DataFrame creation
        total_loss /= total_count

        if self._writer is not None and not is_pretrain:
            # 🧠 ML Signal: Averaging predictions
            # ⚠️ SAST Risk (Low): Potentially large array concatenation
            self._writer.add_scalar("training/loss", total_loss, epoch)

        return total_loss

    def test_epoch(
        self, epoch, data_set, return_pred=False, prefix="test", is_pretrain=False
    ):
        self.model.eval()
        # ⚠️ SAST Risk (Low): Potentially large DataFrame creation
        self.tra.eval()
        # 🧠 ML Signal: Collecting evaluation metrics
        data_set.eval()

        preds = []
        probs = []
        P_all = []
        metrics = []
        # ⚠️ SAST Risk (Low): Potentially large DataFrame creation
        for batch in tqdm(data_set):
            data, state, label, count = (
                batch["data"],
                batch["state"],
                batch["label"],
                batch["daily_count"],
            )
            # ⚠️ SAST Risk (Low): Potentially large DataFrame creation
            index = batch["daily_index"] if self.use_daily_transport else batch["index"]

            with torch.no_grad():
                hidden = self.model(data)
                all_preds, choice, prob = self.tra(hidden, state)

            if is_pretrain or self.transport_method != "none":
                loss, pred, L, P = self.transport_fn(
                    # 🧠 ML Signal: Logging metrics conditionally
                    all_preds,
                    label,
                    choice,
                    prob,
                    state.mean(dim=1),
                    # ⚠️ SAST Risk (Low): Potentially large DataFrame concatenation
                    count,
                    self.transport_method if not is_pretrain else "oracle",
                    self.alpha,
                    training=False,
                    # ✅ Best Practice: Sorting index for consistent ordering
                )
                data_set.assign_data(index, L)  # save loss to memory
                if P is not None and return_pred:
                    # ⚠️ SAST Risk (Low): Potentially large DataFrame concatenation
                    P_all.append(pd.DataFrame(P.cpu().numpy(), index=index))
            else:
                pred = all_preds.mean(dim=1)

            # ✅ Best Practice: Use of logging for tracking the flow and state of the application
            X = np.c_[pred.cpu().numpy(), label.cpu().numpy(), all_preds.cpu().numpy()]
            columns = ["score", "label"] + [
                "score_%d" % d for d in range(all_preds.shape[1])
            ]
            pred = pd.DataFrame(X, index=batch["index"], columns=columns)
            # ✅ Best Practice: Sorting index for consistent ordering

            metrics.append(evaluate(pred))
            # ✅ Best Practice: Use of logging for tracking the flow and state of the application

            # ⚠️ SAST Risk (Low): Potentially large DataFrame concatenation
            if return_pred:
                # ✅ Best Practice: Use of logging for tracking the flow and state of the application
                preds.append(pred)
                if prob is not None:
                    columns = ["prob_%d" % d for d in range(all_preds.shape[1])]
                    # ✅ Best Practice: Use of logging for tracking the flow and state of the application
                    probs.append(
                        pd.DataFrame(prob.cpu().numpy(), index=index, columns=columns)
                    )

        metrics = pd.DataFrame(metrics)
        # ✅ Best Practice: Sorting index for consistent ordering
        metrics = {
            "MSE": metrics.MSE.mean(),
            "MAE": metrics.MAE.mean(),
            # ✅ Best Practice: Use of logging for tracking the flow and state of the application
            "IC": metrics.IC.mean(),
            "ICIR": metrics.IC.mean() / metrics.IC.std(),
        }

        # ✅ Best Practice: Use of logging for tracking the flow and state of the application
        if self._writer is not None and epoch >= 0 and not is_pretrain:
            for key, value in metrics.items():
                self._writer.add_scalar(prefix + "/" + key, value, epoch)

        # ✅ Best Practice: Use of logging for tracking the flow and state of the application
        if return_pred:
            preds = pd.concat(preds, axis=0)
            preds.index = data_set.restore_index(preds.index)
            preds.index = preds.index.swaplevel()
            preds.sort_index(inplace=True)

            if probs:
                probs = pd.concat(probs, axis=0)
                if self.use_daily_transport:
                    probs.index = data_set.restore_daily_index(probs.index)
                else:
                    # ⚠️ SAST Risk (Low): Potential file path manipulation vulnerability
                    probs.index = data_set.restore_index(probs.index)
                    probs.index = probs.index.swaplevel()
                    probs.sort_index(inplace=True)

            if len(P_all):
                # ⚠️ SAST Risk (Low): Using mutable default arguments like dict() can lead to unexpected behavior.
                P_all = pd.concat(P_all, axis=0)
                # ✅ Best Practice: Use of logging for tracking the flow and state of the application
                if self.use_daily_transport:
                    P_all.index = data_set.restore_daily_index(P_all.index)
                else:
                    # ✅ Best Practice: Use of logging for tracking the flow and state of the application
                    P_all.index = data_set.restore_index(P_all.index)
                    P_all.index = P_all.index.swaplevel()
                    P_all.sort_index(inplace=True)

        return metrics, preds, probs, P_all

    # 🧠 ML Signal: Use of Adam optimizer indicates a common pattern in training ML models.

    def _fit(self, train_set, valid_set, test_set, evals_result, is_pretrain=True):
        best_score = -1
        best_epoch = 0
        stop_rounds = 0
        best_params = {
            # 🧠 ML Signal: Re-initializing optimizer for different training phases.
            "model": copy.deepcopy(self.model.state_dict()),
            "tra": copy.deepcopy(self.tra.state_dict()),
        }
        # train
        if not is_pretrain and self.transport_method != "none":
            self.logger.info("init memory...")
            self.test_epoch(-1, train_set)

        for epoch in range(self.n_epochs):
            self.logger.info("Epoch %d:", epoch)

            self.logger.info("training...")
            self.train_epoch(epoch, train_set, is_pretrain=is_pretrain)
            # ⚠️ SAST Risk (Low): Potential path traversal if logdir is not properly sanitized.

            self.logger.info("evaluating...")
            # NOTE: during evaluating, the whole memory will be refreshed
            if not is_pretrain and (
                self.transport_method == "router" or self.eval_train
            ):
                # ⚠️ SAST Risk (Low): Potential path traversal if logdir is not properly sanitized.
                train_set.clear_memory()  # NOTE: clear the shared memory
                train_metrics = self.test_epoch(
                    epoch, train_set, is_pretrain=is_pretrain, prefix="train"
                )[0]
                # ⚠️ SAST Risk (Low): Potential path traversal if logdir is not properly sanitized.
                evals_result["train"].append(train_metrics)
                self.logger.info("train metrics: %s" % train_metrics)
            # ⚠️ SAST Risk (Low): Potential path traversal if logdir is not properly sanitized.

            valid_metrics = self.test_epoch(
                epoch, valid_set, is_pretrain=is_pretrain, prefix="valid"
            )[0]
            # ⚠️ SAST Risk (Low): Potential path traversal if logdir is not properly sanitized.
            evals_result["valid"].append(valid_metrics)
            self.logger.info("valid metrics: %s" % valid_metrics)
            # ⚠️ SAST Risk (Low): Potential path traversal if logdir is not properly sanitized.

            if self.eval_test:
                test_metrics = self.test_epoch(
                    epoch, test_set, is_pretrain=is_pretrain, prefix="test"
                )[0]
                evals_result["test"].append(test_metrics)
                self.logger.info("test metrics: %s" % test_metrics)

            if valid_metrics["IC"] > best_score:
                best_score = valid_metrics["IC"]
                stop_rounds = 0
                best_epoch = epoch
                best_params = {
                    "model": copy.deepcopy(self.model.state_dict()),
                    "tra": copy.deepcopy(self.tra.state_dict()),
                }
                if self.logdir is not None:
                    torch.save(best_params, self.logdir + "/model.bin")
            else:
                stop_rounds += 1
                if stop_rounds >= self.early_stop:
                    self.logger.info("early stop @ %s" % epoch)
                    break

        # ⚠️ SAST Risk (Low): Potential path traversal if logdir is not properly sanitized.
        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.model.load_state_dict(best_params["model"])
        # ✅ Best Practice: Method signature includes default parameter values, improving usability.
        self.tra.load_state_dict(best_params["tra"])

        # ✅ Best Practice: Using assert to enforce type checking for the dataset parameter.
        return best_score

    # ⚠️ SAST Risk (Low): Potential for assert statement to be disabled in optimized mode, leading to type issues.
    def fit(self, dataset, evals_result=dict()):
        # ⚠️ SAST Risk (Medium): Raises a ValueError if the model is not fitted, which could be a denial of service vector if not handled properly by the caller.
        assert isinstance(
            dataset, MTSDatasetH
        ), "TRAModel only supports `qlib.contrib.data.dataset.MTSDatasetH`"

        train_set, valid_set, test_set = dataset.prepare(["train", "valid", "test"])
        # ✅ Best Practice: Class docstring provides a clear description of the class and its parameters
        # ✅ Best Practice: Preparing the dataset for the specified segment.

        # ⚠️ SAST Risk (Low): Potential path traversal if logdir is not properly sanitized.
        # 🧠 ML Signal: Returns predictions, a common pattern in ML model interfaces.
        # 🧠 ML Signal: Calls a method to test the model, indicating a pattern of model evaluation.
        # 🧠 ML Signal: Logging metrics, which is useful for monitoring model performance.
        self.fitted = True
        self.global_step = -1

        evals_result["train"] = []
        evals_result["valid"] = []
        evals_result["test"] = []

        if self.pretrain:
            self.logger.info("pretraining...")
            self.optimizer = optim.Adam(
                list(self.model.parameters()) + list(self.tra.predictors.parameters()),
                lr=self.lr,
            )
            self._fit(train_set, valid_set, test_set, evals_result, is_pretrain=True)

            # reset optimizer
            self.optimizer = optim.Adam(
                list(self.model.parameters()) + list(self.tra.parameters()), lr=self.lr
            )

        self.logger.info("training...")
        best_score = self._fit(
            train_set, valid_set, test_set, evals_result, is_pretrain=False
        )

        self.logger.info("inference")
        train_metrics, train_preds, train_probs, train_P = self.test_epoch(
            -1, train_set, return_pred=True
        )
        self.logger.info("train metrics: %s" % train_metrics)

        valid_metrics, valid_preds, valid_probs, valid_P = self.test_epoch(
            -1, valid_set, return_pred=True
        )
        self.logger.info("valid metrics: %s" % valid_metrics)
        # ✅ Best Practice: Consider validating input parameters for expected types and ranges

        test_metrics, test_preds, test_probs, test_P = self.test_epoch(
            -1, test_set, return_pred=True
        )
        self.logger.info("test metrics: %s" % test_metrics)

        # 🧠 ML Signal: Dynamic RNN architecture selection based on parameter
        # ⚠️ SAST Risk (Low): getattr can lead to security risks if rnn_arch is not validated
        if self.logdir:
            self.logger.info("save model & pred to local directory")

            pd.concat(
                {name: pd.DataFrame(evals_result[name]) for name in evals_result},
                axis=1,
            ).to_csv(self.logdir + "/logs.csv", index=False)

            torch.save(
                {"model": self.model.state_dict(), "tra": self.tra.state_dict()},
                self.logdir + "/model.bin",
            )

            train_preds.to_pickle(self.logdir + "/train_pred.pkl")
            valid_preds.to_pickle(self.logdir + "/valid_pred.pkl")
            test_preds.to_pickle(self.logdir + "/test_pred.pkl")

            if len(train_probs):
                # 🧠 ML Signal: Checks if input projection is used, indicating a flexible model architecture
                train_probs.to_pickle(self.logdir + "/train_prob.pkl")
                valid_probs.to_pickle(self.logdir + "/valid_prob.pkl")
                test_probs.to_pickle(self.logdir + "/test_prob.pkl")
            # 🧠 ML Signal: Use of RNN indicates sequence processing

            if len(train_P):
                # 🧠 ML Signal: Conditional logic based on RNN architecture type
                train_P.to_pickle(self.logdir + "/train_P.pkl")
                valid_P.to_pickle(self.logdir + "/valid_P.pkl")
                test_P.to_pickle(self.logdir + "/test_P.pkl")
            # ✅ Best Practice: Using mean to aggregate outputs for consistent output size

            info = {
                # 🧠 ML Signal: Use of attention mechanism for sequence weighting
                "config": {
                    "model_config": self.model_config,
                    # 🧠 ML Signal: Transformation and non-linearity applied to RNN output
                    # ✅ Best Practice: Inheriting from nn.Module is standard for PyTorch models and layers.
                    "tra_config": self.tra_config,
                    "model_type": self.model_type,
                    # 🧠 ML Signal: Softmax used for attention score calculation
                    # ✅ Best Practice: Use of super() to initialize the parent class
                    "lr": self.lr,
                    "n_epochs": self.n_epochs,
                    # 🧠 ML Signal: Weighted sum for attention output
                    # 🧠 ML Signal: Use of dropout, common in neural network models to prevent overfitting
                    "early_stop": self.early_stop,
                    "max_steps_per_epoch": self.max_steps_per_epoch,
                    # 🧠 ML Signal: Initialization of positional encoding matrix, common in transformer models
                    # ✅ Best Practice: Concatenating outputs for enriched feature representation
                    "lamb": self.lamb,
                    "rho": self.rho,
                    # 🧠 ML Signal: Use of torch.arange to create a sequence of positions
                    "alpha": self.alpha,
                    "seed": self.seed,
                    # 🧠 ML Signal: Calculation of div_term for scaling positions, specific to transformer models
                    "logdir": self.logdir,
                    "pretrain": self.pretrain,
                    # 🧠 ML Signal: Use of sine and cosine functions for positional encoding
                    # 🧠 ML Signal: Use of positional encoding in a forward pass, common in transformer models
                    "init_state": self.init_state,
                    "transport_method": self.transport_method,
                    # 🧠 ML Signal: Use of dropout for regularization in neural networks
                    # 🧠 ML Signal: Definition of a Transformer model class, useful for identifying model architecture patterns
                    "use_daily_transport": self.use_daily_transport,
                    # ✅ Best Practice: Use of register_buffer to store tensors not considered model parameters
                    # 🧠 ML Signal: Reshaping positional encoding for model input
                },
                "best_eval_metric": -best_score,  # NOTE: -1 for minimize
                "metrics": {
                    "train": train_metrics,
                    "valid": valid_metrics,
                    "test": test_metrics,
                },
            }
            with open(self.logdir + "/info.json", "w") as f:
                json.dump(info, f)

    def predict(self, dataset, segment="test"):
        assert isinstance(
            dataset, MTSDatasetH
        ), "TRAModel only supports `qlib.contrib.data.dataset.MTSDatasetH`"

        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        test_set = dataset.prepare(segment)

        metrics, preds, _, _ = self.test_epoch(-1, test_set, return_pred=True)
        self.logger.info("test metrics: %s" % metrics)
        # ✅ Best Practice: Call to super() ensures proper initialization of the base class

        return preds


# 🧠 ML Signal: Storing model hyperparameters as instance variables


# 🧠 ML Signal: Storing model hyperparameters as instance variables
class RNN(nn.Module):
    """RNN Model

    Args:
        input_size (int): input size (# features)
        hidden_size (int): hidden size
        num_layers (int): number of hidden layers
        rnn_arch (str): rnn architecture
        use_attn (bool): whether use attention layer.
            we use concat attention as https://github.com/fulifeng/Adv-ALSTM/
        dropout (float): dropout rate
    # 🧠 ML Signal: Applying positional encoding, a common pattern in transformer models
    """

    # 🧠 ML Signal: Use of input projection, indicating transformation of input features
    def __init__(
        # ✅ Best Practice: Using nn.TransformerEncoder for building the encoder
        # ✅ Best Practice: Importing necessary modules at the beginning of the file
        self,
        # 🧠 ML Signal: Storing model hyperparameters as instance variables
        # 🧠 ML Signal: Use of encoder, typical in sequence-to-sequence models
        # ✅ Best Practice: Returning the last element of the output, common in sequence processing
        input_size=16,
        hidden_size=64,
        num_layers=2,
        rnn_arch="GRU",
        use_attn=True,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__()

        self.input_size = input_size
        # ✅ Best Practice: Using type hints for constructor arguments
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_arch = rnn_arch
        self.use_attn = use_attn

        if hidden_size < input_size:
            # compression
            # ✅ Best Practice: Calling the superclass constructor
            self.input_proj = nn.Linear(input_size, hidden_size)
        else:
            # 🧠 ML Signal: Storing model parameters as instance variables
            self.input_proj = None

        self.rnn = getattr(nn, rnn_arch)(
            # ⚠️ SAST Risk (Low): Use of assert for input validation can be disabled in optimized mode
            input_size=min(input_size, hidden_size),
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            # 🧠 ML Signal: Defining a linear layer for the router
            dropout=dropout,
        )
        # 🧠 ML Signal: Use of nn.Linear indicates a neural network model component
        # 🧠 ML Signal: Defining a linear layer for the output

        if self.use_attn:
            # 🧠 ML Signal: Using ReLU activation function
            # 🧠 ML Signal: Dynamic selection of RNN architecture based on input
            self.W = nn.Linear(hidden_size, hidden_size)
            self.u = nn.Linear(hidden_size, 1, bias=False)
            self.output_size = hidden_size * 2
        else:
            self.output_size = hidden_size

    def forward(self, x):
        # 🧠 ML Signal: Using Gumbel Softmax for sampling
        if self.input_proj is not None:
            x = self.input_proj(x)

        # 🧠 ML Signal: Conditional architecture design based on input parameters
        rnn_out, last_out = self.rnn(x)
        # 🧠 ML Signal: Method to reset parameters, indicating model reinitialization
        if self.rnn_arch == "LSTM":
            last_out = last_out[0]
        # 🧠 ML Signal: Iterating over model components, common in neural network structures
        last_out = last_out.mean(dim=0)

        # 🧠 ML Signal: Recursive parameter reset, typical in hierarchical model structures
        if self.use_attn:
            laten = self.W(rnn_out).tanh()
            scores = self.u(laten).softmax(dim=1)
            att_out = (rnn_out * scores).sum(dim=1)
            last_out = torch.cat([last_out, att_out], dim=1)

        return last_out


class PositionalEncoding(nn.Module):
    # reference: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        # ⚠️ SAST Risk (Low): Use of gumbel_softmax with hard=True can lead to non-differentiable operations
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # ⚠️ SAST Risk (Low): Potential numerical instability in softmax with small tau values

        # 🧠 ML Signal: Function evaluates prediction accuracy using statistical metrics
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 🧠 ML Signal: Ranking predictions is a common preprocessing step in ML
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        # ⚠️ SAST Risk (Low): Assumes 'pred' has 'score' and 'label' attributes
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # ⚠️ SAST Risk (Low): Assumes 'pred' has 'score' and 'label' attributes
        self.register_buffer("pe", pe)

    # 🧠 ML Signal: Calculating difference between predicted and actual values
    def forward(self, x):
        # 🧠 ML Signal: Function to handle infinite values in tensors, useful for preprocessing in ML models
        x = x + self.pe[: x.size(0), :]
        # 🧠 ML Signal: Mean Squared Error is a common metric for regression tasks
        return self.dropout(x)


# 🧠 ML Signal: Identifying infinite values in a tensor, common in data preprocessing
# 🧠 ML Signal: Mean Absolute Error is a common metric for regression tasks


class Transformer(nn.Module):
    """Transformer Model

    Args:
        input_size (int): input size (# features)
        hidden_size (int): hidden size
        num_layers (int): number of transformer layers
        num_heads (int): number of heads in transformer
        dropout (float): dropout rate
    """

    # 🧠 ML Signal: Replacing inf with max value, a common strategy in data normalization
    def __init__(
        self,
        # ✅ Best Practice: Consider adding type hints for function parameters and return type for better readability and maintainability
        input_size=16,
        hidden_size=64,
        # ⚠️ SAST Risk (Low): Using torch.no_grad() suppresses gradient computation, ensure this is intended
        num_layers=2,
        num_heads=2,
        # 🧠 ML Signal: Use of torch.exp suggests this function is part of a machine learning model
        dropout=0.0,
        **kwargs,
        # ✅ Best Practice: Ensure shoot_infs is defined elsewhere in the codebase
    ):
        super().__init__()
        # 🧠 ML Signal: Function for calculating loss, common in training ML models

        # 🧠 ML Signal: Iterative normalization pattern is common in ML algorithms
        self.input_size = input_size
        # ⚠️ SAST Risk (Low): Assumes 'label' is a tensor, potential for runtime errors if not
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 🧠 ML Signal: Handling different tensor shapes, common in ML preprocessing
        self.num_heads = num_heads
        # 🧠 ML Signal: Function for min-max normalization, common in data preprocessing for ML models

        self.input_proj = nn.Linear(input_size, hidden_size)
        # ⚠️ SAST Risk (Low): Assumes 'pred' and 'label' are compatible tensors, potential for runtime errors if not
        # 🧠 ML Signal: Use of tensor operations, indicating potential use in ML frameworks like PyTorch

        self.pe = PositionalEncoding(input_size, dropout)
        # 🧠 ML Signal: Use of tensor operations, indicating potential use in ML frameworks like PyTorch
        layer = nn.TransformerEncoderLayer(
            nhead=num_heads,
            dropout=dropout,
            d_model=hidden_size,
            dim_feedforward=hidden_size * 4,
            # 🧠 ML Signal: Handling edge cases where min equals max, common in data normalization
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        # ⚠️ SAST Risk (Low): Potential division by zero if EPS is not defined or too small
        # 🧠 ML Signal: Handling edge cases where min equals max, common in data normalization
        # ✅ Best Practice: Clear and concise return statement

        self.output_size = hidden_size

    def forward(self, x):
        x = x.permute(1, 0, 2).contiguous()  # the first dim need to be time
        x = self.pe(x)

        x = self.input_proj(x)
        out = self.encoder(x)

        return out[-1]


# ✅ Best Practice: Use of assert statements to validate input shapes and values
class TRA(nn.Module):
    """Temporal Routing Adaptor (TRA)

    TRA takes historical prediction errors & latent representation as inputs,
    then routes the input sample to a specific predictor for training & inference.

    Args:
        input_size (int): input size (RNN/Transformer's hidden size)
        num_states (int): number of latent states (i.e., trading patterns)
            If `num_states=1`, then TRA falls back to traditional methods
        hidden_size (int): hidden size of the router
        tau (float): gumbel softmax temperature
        src_info (str): information for the router
    # 🧠 ML Signal: Combination of current and historical loss with a parameter
    """

    # 🧠 ML Signal: Re-normalization of combined loss
    def __init__(
        self,
        input_size,
        # 🧠 ML Signal: Use of sinkhorn function for transport plan
        num_states=1,
        # ✅ Best Practice: Deleting unused variables to free memory
        hidden_size=8,
        rnn_arch="GRU",
        num_layers=1,
        # ✅ Best Practice: Clear conditional logic for different transport methods
        dropout=0.0,
        tau=1.0,
        # 🧠 ML Signal: Use of weighted sum based on choice tensor
        # 🧠 ML Signal: Use of argmax for selecting predictions
        src_info="LR_TPE",
    ):
        super().__init__()

        assert src_info in ["LR", "TPE", "LR_TPE"], "invalid `src_info`"

        self.num_states = num_states
        self.tau = tau
        self.rnn_arch = rnn_arch
        self.src_info = src_info

        self.predictors = nn.Linear(input_size, num_states)

        # ✅ Best Practice: Clear conditional logic for different transport methods
        # ⚠️ SAST Risk (Low): Lack of input validation for tensor dimensions and types
        # 🧠 ML Signal: Use of weighted sum based on transport plan
        # 🧠 ML Signal: Use of custom loss function
        if self.num_states > 1:
            # 🧠 ML Signal: Calculation of mean loss using transport plan
            if "TPE" in src_info:
                self.router = getattr(nn, rnn_arch)(
                    # 🧠 ML Signal: Returning multiple outputs from a function
                    # ⚠️ SAST Risk (Low): Hardcoded method names could lead to errors if not handled properly
                    input_size=num_states,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout,
                )
                self.fc = nn.Linear(
                    hidden_size + input_size if "LR" in src_info else hidden_size,
                    num_states,
                )
            # ⚠️ SAST Risk (Low): Potential issue if loss_fn is not defined or behaves unexpectedly
            else:
                self.fc = nn.Linear(input_size, num_states)

    # ⚠️ SAST Risk (Low): Assumes all_loss is non-empty and contains valid tensors
    def reset_parameters(self):
        for child in self.children():
            # ✅ Best Practice: Detach tensors to avoid unnecessary gradient tracking
            child.reset_parameters()

    # ✅ Best Practice: Use of alpha for weighted combination is a common pattern
    def forward(self, hidden, hist_loss):
        preds = self.predictors(hidden)

        # ⚠️ SAST Risk (Low): Assumes sinkhorn function is defined and behaves as expected
        if self.num_states == 1:  # no need for router when having only one prediction
            return preds, None, None

        if "TPE" in self.src_info:
            out = self.router(hist_loss)[1]  # TPE
            if self.rnn_arch == "LSTM":
                out = out[0]
            out = out.mean(dim=0)
            if "LR" in self.src_info:
                out = torch.cat([hidden, out], dim=-1)  # LR_TPE
        # 🧠 ML Signal: Use of matrix multiplication for prediction adjustment
        else:
            out = hidden  # LR

        # 🧠 ML Signal: Use of argmax for selecting predictions
        out = self.fc(out)

        choice = F.gumbel_softmax(out, dim=-1, tau=self.tau, hard=True)
        # 🧠 ML Signal: Use of matrix multiplication for prediction adjustment
        prob = torch.softmax(out / self.tau, dim=-1)

        return preds, choice, prob


# ⚠️ SAST Risk (Low): Assumes pred is non-empty and contains valid tensors
# ✅ Best Practice: Initialize lists to collect issues for better error handling and reporting


def evaluate(pred):
    # ⚠️ SAST Risk (Low): Potential issue if loss_fn is not defined or behaves unexpectedly
    pred = pred.rank(pct=True)  # transform into percentiles
    # 🧠 ML Signal: Use of metadata in state_dict indicates handling of model versioning or additional info
    score = pred.score
    label = pred.label
    # ⚠️ SAST Risk (Low): Assumes P and all_loss are compatible for multiplication
    # 🧠 ML Signal: Copying state_dict suggests intention to modify without affecting the original
    diff = score - label
    MSE = (diff**2).mean()
    # ⚠️ SAST Risk (Low): Potential use of undefined variable 'metadata' if not defined elsewhere
    MAE = (diff.abs()).mean()
    # 🧠 ML Signal: Preserving metadata in state_dict indicates importance of additional model information
    # 🧠 ML Signal: Recursive function pattern for loading model components
    IC = score.corr(label, method="spearman")
    return {"MSE": MSE, "MAE": MAE, "IC": IC}


def shoot_infs(inp_tensor):
    """Replaces inf by maximum of tensor"""
    mask_inf = torch.isinf(inp_tensor)
    # 🧠 ML Signal: Recursive call to handle nested modules
    ind_inf = torch.nonzero(mask_inf, as_tuple=False)
    if len(ind_inf) > 0:
        # 🧠 ML Signal: Function call to load model state
        # ⚠️ SAST Risk (Low): No import statements for pd, plt, io, np; potential NameError if not imported
        for ind in ind_inf:
            if len(ind) == 2:
                # ⚠️ SAST Risk (Low): Overwriting the 'load' function with None, which can lead to errors if 'load' is called again
                # ✅ Best Practice: Use isinstance to check if P is a DataFrame
                inp_tensor[ind[0], ind[1]] = 0
            elif len(ind) == 1:
                # ✅ Best Practice: Returning a dictionary for structured error reporting
                # ✅ Best Practice: Use of subplots for multiple plots in a single figure
                inp_tensor[ind[0]] = 0
        m = torch.max(inp_tensor)
        # 🧠 ML Signal: Plotting area chart, common in data visualization tasks
        for ind in ind_inf:
            if len(ind) == 2:
                # 🧠 ML Signal: Using idxmax and value_counts for data analysis
                inp_tensor[ind[0], ind[1]] = m
            elif len(ind) == 1:
                # ✅ Best Practice: Use of tight_layout to prevent overlap of subplots
                inp_tensor[ind[0]] = m
    # ✅ Best Practice: Use of BytesIO for in-memory byte buffer
    # ✅ Best Practice: Save figure to buffer in PNG format
    # ✅ Best Practice: Read image from buffer
    # ✅ Best Practice: Close the plot to free up resources
    # 🧠 ML Signal: Conversion of image data to uint8 format, common in image processing
    return inp_tensor


def sinkhorn(Q, n_iters=3, epsilon=0.1):
    # epsilon should be adjusted according to logits value's scale
    with torch.no_grad():
        Q = torch.exp(Q / epsilon)
        Q = shoot_infs(Q)
        for i in range(n_iters):
            Q /= Q.sum(dim=0, keepdim=True)
            Q /= Q.sum(dim=1, keepdim=True)
    return Q


def loss_fn(pred, label):
    mask = ~torch.isnan(label)
    if len(pred.shape) == 2:
        label = label[:, None]
    return (pred[mask] - label[mask]).pow(2).mean(dim=0)


def minmax_norm(x):
    xmin = x.min(dim=-1, keepdim=True).values
    xmax = x.max(dim=-1, keepdim=True).values
    mask = (xmin == xmax).squeeze()
    x = (x - xmin) / (xmax - xmin + EPS)
    x[mask] = 1
    return x


def transport_sample(
    all_preds,
    label,
    choice,
    prob,
    hist_loss,
    count,
    transport_method,
    alpha,
    training=False,
):
    """
    sample-wise transport

    Args:
        all_preds (torch.Tensor): predictions from all predictors, [sample x states]
        label (torch.Tensor): label, [sample]
        choice (torch.Tensor): gumbel softmax choice, [sample x states]
        prob (torch.Tensor): router predicted probility, [sample x states]
        hist_loss (torch.Tensor): history loss matrix, [sample x states]
        count (list): sample counts for each day, empty list for sample-wise transport
        transport_method (str): transportation method
        alpha (float): fusion parameter for calculating transport loss matrix
        training (bool): indicate training or inference
    """
    assert all_preds.shape == choice.shape
    assert len(all_preds) == len(label)
    assert transport_method in ["oracle", "router"]

    all_loss = torch.zeros_like(all_preds)
    mask = ~torch.isnan(label)
    all_loss[mask] = (all_preds[mask] - label[mask, None]).pow(2)  # [sample x states]

    L = minmax_norm(all_loss.detach())
    Lh = L * alpha + minmax_norm(hist_loss) * (1 - alpha)  # add hist loss for transport
    Lh = minmax_norm(Lh)
    P = sinkhorn(-Lh)
    del Lh

    if transport_method == "router":
        if training:
            pred = (all_preds * choice).sum(dim=1)  # gumbel softmax
        else:
            pred = all_preds[range(len(all_preds)), prob.argmax(dim=-1)]  # argmax
    else:
        pred = (all_preds * P).sum(dim=1)

    if transport_method == "router":
        loss = loss_fn(pred, label)
    else:
        loss = (all_loss * P).sum(dim=1).mean()

    return loss, pred, L, P


def transport_daily(
    all_preds,
    label,
    choice,
    prob,
    hist_loss,
    count,
    transport_method,
    alpha,
    training=False,
):
    """
    daily transport

    Args:
        all_preds (torch.Tensor): predictions from all predictors, [sample x states]
        label (torch.Tensor): label, [sample]
        choice (torch.Tensor): gumbel softmax choice, [days x states]
        prob (torch.Tensor): router predicted probility, [days x states]
        hist_loss (torch.Tensor): history loss matrix, [days x states]
        count (list): sample counts for each day, [days]
        transport_method (str): transportation method
        alpha (float): fusion parameter for calculating transport loss matrix
        training (bool): indicate training or inference
    """
    assert len(prob) == len(count)
    assert len(all_preds) == sum(count)
    assert transport_method in ["oracle", "router"]

    all_loss = []  # loss of all predictions
    start = 0
    for i, cnt in enumerate(count):
        slc = slice(start, start + cnt)  # samples from the i-th day
        start += cnt
        tloss = loss_fn(all_preds[slc], label[slc])  # loss of the i-th day
        all_loss.append(tloss)
    all_loss = torch.stack(all_loss, dim=0)  # [days x states]

    L = minmax_norm(all_loss.detach())
    Lh = L * alpha + minmax_norm(hist_loss) * (1 - alpha)  # add hist loss for transport
    Lh = minmax_norm(Lh)
    P = sinkhorn(-Lh)
    del Lh

    pred = []
    start = 0
    for i, cnt in enumerate(count):
        slc = slice(start, start + cnt)  # samples from the i-th day
        start += cnt
        if transport_method == "router":
            if training:
                tpred = all_preds[slc] @ choice[i]  # gumbel softmax
            else:
                tpred = all_preds[slc][:, prob[i].argmax(dim=-1)]  # argmax
        else:
            tpred = all_preds[slc] @ P[i]
        pred.append(tpred)
    pred = torch.cat(pred, dim=0)  # [samples]

    if transport_method == "router":
        loss = loss_fn(pred, label)
    else:
        loss = (all_loss * P).sum(dim=1).mean()

    return loss, pred, L, P


def load_state_dict_unsafe(model, state_dict):
    """
    Load state dict to provided model while ignore exceptions.
    """

    missing_keys = []
    unexpected_keys = []
    error_msgs = []

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            True,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(model)
    load = None  # break load->load reference cycle

    return {
        "unexpected_keys": unexpected_keys,
        "missing_keys": missing_keys,
        "error_msgs": error_msgs,
    }


def plot(P):
    assert isinstance(P, pd.DataFrame)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    P.plot.area(ax=axes[0], xlabel="")
    P.idxmax(axis=1).value_counts().sort_index().plot.bar(ax=axes[1], xlabel="")
    plt.tight_layout()

    with io.BytesIO() as buf:
        plt.savefig(buf, format="png")
        buf.seek(0)
        img = plt.imread(buf)
        plt.close()

    return np.uint8(img * 255)
