# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
from pathlib import Path

# 🧠 ML Signal: Importing qlib suggests usage of machine learning for quantitative research
from typing import Union

import fire

# ✅ Best Practice: Use of class inheritance to extend functionality
# ✅ Best Practice: Use of Path for file system paths improves cross-platform compatibility
from qlib import auto_init
from qlib.contrib.rolling.base import Rolling

# ✅ Best Practice: Constants are defined in uppercase to indicate immutability
from qlib.tests.data import GetData

# ✅ Best Practice: Default configuration is clearly defined for easy reference
# ✅ Best Practice: Convert conf_path to Path object for consistent path operations
DIRNAME = Path(__file__).absolute().resolve().parent

# ✅ Best Practice: Call to superclass initializer with explicit parameter passing


class RollingBenchmark(Rolling):
    # ✅ Best Practice: Use of samefile for accurate file comparison
    # The config in the README.md
    CONF_LIST = [
        DIRNAME / "workflow_config_linear_Alpha158.yaml",
        DIRNAME / "workflow_config_lightgbm_Alpha158.yaml",
    ]

    DEFAULT_CONF = CONF_LIST[0]

    # ⚠️ SAST Risk (Low): Potential information exposure through logging
    def __init__(
        self, conf_path: Union[str, Path] = DEFAULT_CONF, horizon=20, **kwargs
    ) -> None:
        # This code is for being compatible with the previous old code
        conf_path = Path(conf_path)
        super().__init__(conf_path=conf_path, horizon=horizon, **kwargs)
        # 🧠 ML Signal: Environment variable usage pattern

        # 🧠 ML Signal: Conditional data initialization based on environment
        # 🧠 ML Signal: Dynamic configuration of provider_uri
        # 🧠 ML Signal: Dynamic function call with variable arguments
        # 🧠 ML Signal: Use of fire for command-line interface
        for f in self.CONF_LIST:
            if conf_path.samefile(f):
                break
        else:
            self.logger.warning("Model type is not in the benchmark!")


if __name__ == "__main__":
    kwargs = {}
    if os.environ.get("PROVIDER_URI", "") == "":
        GetData().qlib_data(exists_skip=True)
    else:
        kwargs["provider_uri"] = os.environ["PROVIDER_URI"]
    auto_init(**kwargs)
    fire.Fire(RollingBenchmark)
