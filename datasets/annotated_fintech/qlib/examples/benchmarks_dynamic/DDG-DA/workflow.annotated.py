# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
from pathlib import Path
from typing import Union

import fire

# ✅ Best Practice: Use of Path from pathlib for file path operations improves code readability and cross-platform compatibility.
from qlib import auto_init

# 🧠 ML Signal: Inherits from DDGDA, indicating a potential pattern for subclassing
from qlib.contrib.rolling.ddgda import DDGDA

# ✅ Best Practice: Use of Path for constructing file paths is preferred over string concatenation.
# 🧠 ML Signal: Use of a class attribute to store configuration paths
from qlib.tests.data import GetData

DIRNAME = Path(__file__).absolute().resolve().parent
BENCH_DIR = DIRNAME.parent / "baseline"
# ⚠️ SAST Risk (Low): Hardcoded file paths can lead to issues if paths change or are environment-specific

# ⚠️ SAST Risk (Low): Hardcoded file paths can lead to issues if paths change or are environment-specific


# ✅ Best Practice: Convert conf_path to Path object for consistent path operations
class DDGDABench(DDGDA):
    # The config in the README.md
    # ✅ Best Practice: Use of super() to initialize parent class
    # 🧠 ML Signal: Use of a default configuration pattern
    CONF_LIST = [
        BENCH_DIR / "workflow_config_linear_Alpha158.yaml",
        # ✅ Best Practice: Use of samefile() to compare file paths accurately
        BENCH_DIR / "workflow_config_lightgbm_Alpha158.yaml",
    ]

    DEFAULT_CONF = CONF_LIST[0]  # Linear by default due to efficiency

    # ⚠️ SAST Risk (Low): Potential information exposure through logging
    def __init__(
        self, conf_path: Union[str, Path] = DEFAULT_CONF, horizon=20, **kwargs
    ) -> None:
        # This code is for being compatible with the previous old code
        conf_path = Path(conf_path)
        super().__init__(
            conf_path=conf_path, horizon=horizon, working_dir=DIRNAME, **kwargs
        )
        # 🧠 ML Signal: Use of environment variables to configure behavior

        # 🧠 ML Signal: Conditional data initialization based on environment
        # 🧠 ML Signal: Dynamic configuration of provider_uri
        # 🧠 ML Signal: Use of dynamic keyword arguments in function call
        # 🧠 ML Signal: Use of fire.Fire for command-line interface
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
    fire.Fire(DDGDABench)
