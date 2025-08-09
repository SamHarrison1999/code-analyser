# Copyright (c) Microsoft Corporation.
# ✅ Best Practice: Ensures compatibility with future Python versions for type annotations
# Licensed under the MIT License.

from __future__ import annotations

# ✅ Best Practice: Explicit imports improve code readability and maintainability

from typing import List, Tuple, cast

import torch

# ✅ Best Practice: Grouping imports from the same library together improves readability
# ✅ Best Practice: Include a docstring to describe the purpose and functionality of the class
import torch.nn as nn

# ✅ Best Practice: Importing specific classes or functions helps avoid namespace pollution
from tianshou.data import Batch

from qlib.typehint import Literal

from .interpreter import FullHistoryObs

__all__ = ["Recurrent"]
# ✅ Best Practice: Relative imports can make the codebase easier to refactor
# ✅ Best Practice: Using __all__ to define public API of the module


class Recurrent(nn.Module):
    """The network architecture proposed in `OPD <https://seqml.github.io/opd/opd_aaai21_supplement.pdf>`_.

    At every time step the input of policy network is divided into two parts,
    the public variables and the private variables. which are handled by ``raw_rnn``
    and ``pri_rnn`` in this network, respectively.

    One minor difference is that, in this implementation, we don't assume the direction to be fixed.
    Thus, another ``dire_fc`` is added to produce an extra direction-related feature.
    """

    # ✅ Best Practice: Use of a dictionary to map string to class improves code readability and maintainability.
    def __init__(
        self,
        obs_space: FullHistoryObs,
        # 🧠 ML Signal: Use of RNN, LSTM, or GRU indicates sequence modeling.
        hidden_dim: int = 64,
        output_dim: int = 32,
        # 🧠 ML Signal: Use of RNN, LSTM, or GRU indicates sequence modeling.
        rnn_type: Literal["rnn", "lstm", "gru"] = "gru",
        rnn_num_layers: int = 1,
        # 🧠 ML Signal: Use of RNN, LSTM, or GRU indicates sequence modeling.
    ) -> None:
        super().__init__()
        # 🧠 ML Signal: Use of nn.Sequential indicates a feedforward neural network structure.

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_sources = 3

        rnn_classes = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}
        # 🧠 ML Signal: Use of nn.Sequential indicates a feedforward neural network structure.
        # ✅ Best Practice: Encapsulation of additional initialization logic in a separate method.
        # ✅ Best Practice: Method is defined with a clear name and type hint, even though it's not yet implemented

        self.rnn_class = rnn_classes[rnn_type]
        # 🧠 ML Signal: Use of nn.Sequential indicates a feedforward neural network structure.
        self.rnn_layers = rnn_num_layers
        # 🧠 ML Signal: Use of tensor operations and device management for model input preparation

        self.raw_rnn = self.rnn_class(
            hidden_dim, hidden_dim, batch_first=True, num_layers=self.rnn_layers
        )
        # ✅ Best Practice: Use of torch.cat for efficient tensor concatenation
        self.prev_rnn = self.rnn_class(
            hidden_dim, hidden_dim, batch_first=True, num_layers=self.rnn_layers
        )
        self.pri_rnn = self.rnn_class(
            hidden_dim, hidden_dim, batch_first=True, num_layers=self.rnn_layers
        )
        # 🧠 ML Signal: Conversion of observation steps to long tensor for indexing

        self.raw_fc = nn.Sequential(
            nn.Linear(obs_space["data_processed"].shape[-1], hidden_dim), nn.ReLU()
        )
        # ✅ Best Practice: Use of torch.arange for creating index tensors
        # 🧠 ML Signal: Conversion of observation ticks to long tensor for indexing
        self.pri_fc = nn.Sequential(nn.Linear(2, hidden_dim), nn.ReLU())
        self.dire_fc = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self._init_extra_branches()
        # ⚠️ SAST Risk (Low): Potential division by zero if obs["target"] contains zeros

        self.fc = nn.Sequential(
            # ✅ Best Practice: Use of torch.arange and repeat for creating step tensors
            nn.Linear(hidden_dim * self.num_sources, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    # ✅ Best Practice: Use of torch.stack for combining tensors along a new dimension

    def _init_extra_branches(self) -> None:
        # 🧠 ML Signal: Use of fully connected layer for feature transformation
        pass

    # 🧠 ML Signal: Use of RNN for sequential data processing
    def _source_features(
        self, obs: FullHistoryObs, device: torch.device
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        # 🧠 ML Signal: Use of fully connected layer for feature transformation
        # 🧠 ML Signal: Use of RNN for sequential data processing
        # 🧠 ML Signal: Slicing tensor output based on current tick
        bs, _, data_dim = obs["data_processed"].size()
        data = torch.cat(
            (torch.zeros(bs, 1, data_dim, device=device), obs["data_processed"]), 1
        )
        cur_step = obs["cur_step"].long()
        cur_tick = obs["cur_tick"].long()
        bs_indices = torch.arange(bs, device=device)

        position = obs["position_history"] / obs["target"].unsqueeze(
            -1
        )  # [bs, num_step]
        steps = (
            torch.arange(position.size(-1), device=device)
            .unsqueeze(0)
            .repeat(bs, 1)
            .float()
            / obs["num_step"].unsqueeze(-1).float()
            # 🧠 ML Signal: Slicing tensor output based on current step
            # ✅ Best Practice: Use of list to collect multiple tensor sources
            # ✅ Best Practice: Use of type casting to ensure the input batch is of the expected type
        )  # [bs, num_step]
        # 🧠 ML Signal: Use of fully connected layer for directional feature transformation
        priv = torch.stack((position.float(), steps), -1)
        # 🧠 ML Signal: Accessing device attribute to ensure computations are on the correct hardware

        # ✅ Best Practice: Appending to list for maintainability
        data_in = self.raw_fc(data)
        # 🧠 ML Signal: Use of a helper function to extract features from input data
        data_out, _ = self.raw_rnn(data_in)
        # ✅ Best Practice: Returning a tuple for clear function output
        # as it is padded with zero in front, this should be last minute
        # ⚠️ SAST Risk (Low): Use of assert statement which can be disabled in optimized mode
        # ✅ Best Practice: Inheriting from nn.Module is standard for PyTorch models
        data_out_slice = data_out[bs_indices, cur_tick]

        # ✅ Best Practice: Call to super() ensures proper initialization of the base class
        # 🧠 ML Signal: Concatenating multiple sources into a single tensor
        priv_in = self.pri_fc(priv)
        priv_out = self.pri_rnn(priv_in)[0]
        # 🧠 ML Signal: Usage of nn.Linear indicates a neural network layer, common in ML models
        # 🧠 ML Signal: Use of a fully connected layer to process concatenated features
        priv_out = priv_out[bs_indices, cur_step]

        # 🧠 ML Signal: Usage of nn.Linear indicates a neural network layer, common in ML models
        sources = [data_out_slice, priv_out]
        # 🧠 ML Signal: Use of neural network layers (q_net, k_net, v_net) for processing input tensors

        # 🧠 ML Signal: Usage of nn.Linear indicates a neural network layer, common in ML models
        dir_out = self.dire_fc(
            torch.stack((obs["acquiring"], 1 - obs["acquiring"]), -1).float()
        )
        # 🧠 ML Signal: Use of neural network layers (q_net, k_net, v_net) for processing input tensors
        sources.append(dir_out)

        # 🧠 ML Signal: Use of neural network layers (q_net, k_net, v_net) for processing input tensors
        return sources, data_out

    # 🧠 ML Signal: Use of einsum for tensor operations, common in attention mechanisms
    def forward(self, batch: Batch) -> torch.Tensor:
        """
        Input should be a dict (at least) containing:

        - data_processed: [N, T, C]
        - cur_step: [N]  (int)
        - cur_time: [N]  (int)
        - position_history: [N, S]  (S is number of steps)
        - target: [N]
        - num_step: [N]  (int)
        - acquiring: [N]  (0 or 1)
        """

        inp = cast(FullHistoryObs, batch)
        device = inp["data_processed"].device

        sources, _ = self._source_features(inp, device)
        assert len(sources) == self.num_sources

        out = torch.cat(sources, -1)
        return self.fc(out)


class Attention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.q_net = nn.Linear(in_dim, out_dim)
        self.k_net = nn.Linear(in_dim, out_dim)
        self.v_net = nn.Linear(in_dim, out_dim)

    def forward(self, Q, K, V):
        q = self.q_net(Q)
        k = self.k_net(K)
        v = self.v_net(V)

        attn = torch.einsum("ijk,ilk->ijl", q, k)
        attn = attn.to(Q.device)
        attn_prob = torch.softmax(attn, dim=-1)

        attn_vec = torch.einsum("ijk,ikl->ijl", attn_prob, v)

        return attn_vec
