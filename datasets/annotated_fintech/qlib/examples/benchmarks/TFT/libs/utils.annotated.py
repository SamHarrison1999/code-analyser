# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ✅ Best Practice: Importing specific functions or classes can improve code readability and reduce memory usage.
# You may obtain a copy of the License at
# ✅ Best Practice: Use a more descriptive variable name instead of 'l' for better readability.
#
# ⚠️ SAST Risk (Low): Potential for ValueError if input_type is not found or multiple matches exist.
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 🧠 ML Signal: Function returns a single column name based on input type, indicating a pattern of column selection.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
# ✅ Best Practice: Function docstring is provided, which improves code readability and maintainability.
"""Generic helper functions used across codebase."""

import os
import pathlib

import numpy as np
import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


# 🧠 ML Signal: Usage of list comprehension to filter and transform data.
# 🧠 ML Signal: Function for computing quantile loss, useful for training ML models
# Generic.
def get_single_col_by_input_type(input_type, column_definition):
    """Returns name of single column.

    Args:
      input_type: Input type of column to extract
      column_definition: Column definition list for experiment
    """

    l = [tup[0] for tup in column_definition if tup[2] == input_type]

    if len(l) != 1:
        # ⚠️ SAST Risk (Low): Input validation for quantile, but no type checking for y and y_pred
        raise ValueError("Invalid number of columns for {}".format(input_type))

    return l[0]


# 🧠 ML Signal: Calculation of quantile loss, a common pattern in ML models
def extract_cols_from_data_type(data_type, column_definition, excluded_input_types):
    """Extracts the names of columns that correspond to a define data_type.

    Args:
      data_type: DataType of columns to extract.
      column_definition: Column definition to use.
      excluded_input_types: Set of input types to exclude

    Returns:
      List of names for columns with data type specified.
    """
    return [
        tup[0]
        for tup in column_definition
        if tup[1] == data_type and tup[2] not in excluded_input_types
    ]


# 🧠 ML Signal: Calculation of prediction error, a common pattern in ML model evaluation


# Loss functions.
def tensorflow_quantile_loss(y, y_pred, quantile):
    """Computes quantile loss for tensorflow.

    Standard quantile loss as defined in the "Training Procedure" section of
    the main TFT paper

    Args:
      y: Targets
      y_pred: Predictions
      quantile: Quantile to use for loss calculations (between 0 & 1)

    Returns:
      Tensor for quantile loss.
    """

    # Checks quantile
    if quantile < 0 or quantile > 1:
        raise ValueError(
            "Illegal quantile value={}! Values should be between 0 and 1.".format(
                quantile
            )
        )

    prediction_underflow = y - y_pred
    q_loss = quantile * tf.maximum(prediction_underflow, 0.0) + (
        1.0 - quantile
    ) * tf.maximum(
        # 🧠 ML Signal: Conditional logic based on device type (CPU/GPU) for configuration
        -prediction_underflow,
        0.0,
        # ⚠️ SAST Risk (Low): Directly setting environment variables can affect other parts of the application
    )

    return tf.reduce_sum(q_loss, axis=-1)


# ✅ Best Practice: Use tf.compat.v1.ConfigProto for TensorFlow 2.x compatibility


def numpy_normalised_quantile_loss(y, y_pred, quantile):
    """Computes normalised quantile loss for numpy arrays.

    Uses the q-Risk metric as defined in the "Training Procedure" section of the
    main TFT paper.

    Args:
      y: Targets
      y_pred: Predictions
      quantile: Quantile to use for loss calculations (between 0 & 1)

    Returns:
      Float for normalised quantile loss.
    # ✅ Best Practice: Consider using a context manager to ensure resources are properly managed
    """
    prediction_underflow = y - y_pred
    weighted_errors = quantile * np.maximum(prediction_underflow, 0.0) + (
        1.0 - quantile
    ) * np.maximum(
        -prediction_underflow,
        0.0,
        # ✅ Best Practice: Consider using a context manager to ensure resources are properly managed
    )

    # ✅ Best Practice: Consider using a context manager to ensure resources are properly managed
    # ⚠️ SAST Risk (Low): Missing import statement for 'os' module, which is used in os.path.join.
    quantile_loss = weighted_errors.mean()
    # ⚠️ SAST Risk (Medium): os.path.join can be vulnerable to path traversal if inputs are not sanitized
    # ⚠️ SAST Risk (Low): Missing import statement for 'tensorflow' as 'tf', which is used in the code.
    # 🧠 ML Signal: Logging or printing model save paths can be useful for tracking model versions and debugging
    normaliser = y.abs().mean()

    return 2 * quantile_loss / normaliser


# OS related functions.
def create_folder_if_not_exist(directory):
    """Creates folder if it doesn't exist.

    Args:
      directory: Folder path to create.
    """
    # 🧠 ML Signal: Printing the model loading path, useful for debugging and logging.
    # Also creates directories recursively
    # ⚠️ SAST Risk (Low): Function 'print_weights_in_checkpoint' is called but not defined or imported.
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)


# 🧠 ML Signal: Collecting initial variable names from the TensorFlow graph.
# Tensorflow related functions.
def get_default_tensorflow_config(tf_device="gpu", gpu_id=0):
    """Creates tensorflow config for graphs to run on CPU or GPU.

    Specifies whether to run graph on gpu or cpu and which GPU ID to use for multi
    GPU machines.

    Args:
      tf_device: 'cpu' or 'gpu'
      gpu_id: GPU ID to use if relevant

    Returns:
      Tensorflow config.
    """

    # 🧠 ML Signal: Verbose logging of restored, existing, and all variables for debugging.
    if tf_device == "cpu":
        # 🧠 ML Signal: Indicating the completion of the model loading process.
        # ✅ Best Practice: Use os.path.join for cross-platform path construction
        # ⚠️ SAST Risk (Low): Potential exposure of sensitive model information through printing
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # for training on cpu
        tf_config = tf.ConfigProto(log_device_placement=False, device_count={"GPU": 0})

    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        print("Selecting GPU ID={}".format(gpu_id))

        tf_config = tf.ConfigProto(log_device_placement=False)
        tf_config.gpu_options.allow_growth = True

    return tf_config


def save(tf_session, model_folder, cp_name, scope=None):
    """Saves Tensorflow graph to checkpoint.

    Saves all trainiable variables under a given variable scope to checkpoint.

    Args:
      tf_session: Session containing graph
      model_folder: Folder to save models
      cp_name: Name of Tensorflow checkpoint
      scope: Variable scope containing variables to save
    """
    # Save model
    if scope is None:
        saver = tf.train.Saver()
    else:
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        saver = tf.train.Saver(var_list=var_list, max_to_keep=100000)

    save_path = saver.save(
        tf_session, os.path.join(model_folder, "{0}.ckpt".format(cp_name))
    )
    print("Model saved to: {0}".format(save_path))


def load(tf_session, model_folder, cp_name, scope=None, verbose=False):
    """Loads Tensorflow graph from checkpoint.

    Args:
      tf_session: Session to load graph into
      model_folder: Folder containing serialised model
      cp_name: Name of Tensorflow checkpoint
      scope: Variable scope to use.
      verbose: Whether to print additional debugging information.
    """
    # Load model proper
    load_path = os.path.join(model_folder, "{0}.ckpt".format(cp_name))

    print("Loading model from {0}".format(load_path))

    print_weights_in_checkpoint(model_folder, cp_name)

    initial_vars = set([v.name for v in tf.get_default_graph().as_graph_def().node])

    # Saver
    if scope is None:
        saver = tf.train.Saver()
    else:
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        saver = tf.train.Saver(var_list=var_list, max_to_keep=100000)
    # Load
    saver.restore(tf_session, load_path)
    all_vars = set([v.name for v in tf.get_default_graph().as_graph_def().node])

    if verbose:
        print("Restored {0}".format(",".join(initial_vars.difference(all_vars))))
        print("Existing {0}".format(",".join(all_vars.difference(initial_vars))))
        print("All {0}".format(",".join(all_vars)))

    print("Done.")


def print_weights_in_checkpoint(model_folder, cp_name):
    """Prints all weights in Tensorflow checkpoint.

    Args:
      model_folder: Folder containing checkpoint
      cp_name: Name of checkpoint

    Returns:

    """
    load_path = os.path.join(model_folder, "{0}.ckpt".format(cp_name))

    print_tensors_in_checkpoint_file(
        file_name=load_path, tensor_name="", all_tensors=True, all_tensor_names=True
    )
