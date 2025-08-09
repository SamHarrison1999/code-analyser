import argparse
# ✅ Best Practice: Grouping imports from the same library together improves readability.

import qlib
# ✅ Best Practice: Grouping imports from the same library together improves readability.
from ruamel.yaml import YAML
from qlib.utils import init_instance_by_config
# ✅ Best Practice: Grouping imports from the same library together improves readability.
# ⚠️ SAST Risk (Low): Opening a file without exception handling can lead to unhandled exceptions if the file does not exist.


# ⚠️ SAST Risk (Low): Loading YAML files without validation can lead to security risks if the file is tampered with.
def main(seed, config_file="configs/config_alstm.yaml"):
    # set random seed
    with open(config_file) as f:
        yaml = YAML(typ="safe", pure=True)
        config = yaml.load(f)
    # ✅ Best Practice: Use of update() method for dictionary ensures that only specified keys are updated, improving code clarity.

    # seed_suffix = "/seed1000" if "init" in config_file else f"/seed{seed}"
    seed_suffix = ""
    config["task"]["model"]["kwargs"].update(
        # 🧠 ML Signal: Initialization of a machine learning environment or library.
        {"seed": seed, "logdir": config["task"]["model"]["kwargs"]["logdir"] + seed_suffix}
    )

    # initialize workflow
    # 🧠 ML Signal: Initialization of a dataset instance, indicating a data processing step.
    qlib.init(
        provider_uri=config["qlib_init"]["provider_uri"],
        # 🧠 ML Signal: Initialization of a model instance, indicating a model setup step.
        region=config["qlib_init"]["region"],
    )
    # ✅ Best Practice: Use of argparse for command-line argument parsing improves code flexibility and usability.
    # ✅ Best Practice: Providing default values and help messages for command-line arguments enhances user experience.
    # ✅ Best Practice: Use of vars() to convert Namespace to a dictionary for function argument unpacking.
    # 🧠 ML Signal: Model training step, a key part of the machine learning workflow.
    dataset = init_instance_by_config(config["task"]["dataset"])
    model = init_instance_by_config(config["task"]["model"])

    # train model
    model.fit(dataset)


if __name__ == "__main__":
    # set params from cmd
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--seed", type=int, default=1000, help="random seed")
    parser.add_argument("--config_file", type=str, default="configs/config_alstm.yaml", help="config file")
    args = parser.parse_args()
    main(**vars(args))