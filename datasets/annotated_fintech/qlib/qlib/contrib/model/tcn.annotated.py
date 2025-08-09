# MIT License
# ✅ Best Practice: Import only the necessary functions or classes to keep the namespace clean
# Copyright (c) 2018 CMU Locus Lab
# ⚠️ SAST Risk (Low): Inherits from nn.Module, ensure proper initialization and usage of PyTorch modules
import torch.nn as nn
from torch.nn.utils import weight_norm

# 🧠 ML Signal: Constructor method, often used to initialize class attributes


# ✅ Best Practice: Method should have a docstring explaining its purpose and parameters
# ✅ Best Practice: Explicitly calling the superclass constructor
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        # 🧠 ML Signal: Storing parameter as an instance attribute
        # 🧠 ML Signal: Use of slicing to manipulate tensor dimensions
        # ✅ Best Practice: Inheriting from nn.Module is standard for defining custom neural network layers in PyTorch.
        super(Chomp1d, self).__init__()
        # ⚠️ SAST Risk (Low): Potential for IndexError if chomp_size is larger than the dimension size
        self.chomp_size = chomp_size

    # 🧠 ML Signal: Use of weight normalization in neural network layers
    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


# 🧠 ML Signal: Custom layer for sequence data processing
class TemporalBlock(nn.Module):
    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2
    ):
        # 🧠 ML Signal: Use of ReLU activation function
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
            # 🧠 ML Signal: Use of dropout for regularization
        )
        # 🧠 ML Signal: Use of weight normalization in neural network layers
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            # 🧠 ML Signal: Custom layer for sequence data processing
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
            # 🧠 ML Signal: Use of ReLU activation function
        )
        self.chomp2 = Chomp1d(padding)
        # 🧠 ML Signal: Use of dropout for regularization
        self.relu2 = nn.ReLU()
        # 🧠 ML Signal: Custom weight initialization pattern for neural network layers
        self.dropout2 = nn.Dropout(dropout)
        # 🧠 ML Signal: Sequential model construction

        # 🧠 ML Signal: Custom weight initialization pattern for neural network layers
        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
            # ✅ Best Practice: Check for None before accessing attributes to avoid runtime errors
        )
        # ✅ Best Practice: Conditional logic for layer creation
        # 🧠 ML Signal: Use of a forward method suggests this is part of a neural network model
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        # 🧠 ML Signal: Custom weight initialization pattern for neural network layers
        self.relu = nn.ReLU()
        # 🧠 ML Signal: Use of ReLU activation function
        # 🧠 ML Signal: Use of residual connections is common in deep learning models
        self.init_weights()

    # 🧠 ML Signal: Custom neural network class definition

    # ✅ Best Practice: Initialization of model weights
    # ✅ Best Practice: Use of relu activation function is a common practice in neural networks
    def init_weights(self):
        # ✅ Best Practice: Call to super() ensures proper initialization of the parent class
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            # 🧠 ML Signal: Use of num_channels to determine the number of levels in the network
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # 🧠 ML Signal: Use of exponential growth for dilation size
        out = self.net(x)
        # 🧠 ML Signal: Conditional logic to determine in_channels based on layer index
        # 🧠 ML Signal: Use of num_channels to determine out_channels for each layer
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            # 🧠 ML Signal: Calculation of padding based on kernel_size and dilation
            # 🧠 ML Signal: Method named 'forward' suggests this is a neural network model component
            out_channels = num_channels[i]
            # ✅ Best Practice: Use of nn.Sequential to manage layers in a neural network
            # 🧠 ML Signal: Usage of 'self.network' indicates a class attribute, likely a neural network layer or model
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
