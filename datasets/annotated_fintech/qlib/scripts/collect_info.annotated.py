import sys
import platform
import qlib
import fire
import pkg_resources
from pathlib import Path

# ✅ Best Practice: Use of Path from pathlib for file path operations improves readability and cross-platform compatibility.
QLIB_PATH = Path(__file__).absolute().resolve().parent.parent


class InfoCollector:
    """
    User could collect system info by following commands
    `cd scripts && python collect_info.py all`
    - NOTE: please avoid running this script in the project folder which contains `qlib`
    # 🧠 ML Signal: Iterating over a list of method names to dynamically call them
    """

    # ⚠️ SAST Risk (Low): Using getattr without validation can lead to calling unintended methods
    # ✅ Best Practice: Method docstring provides a brief description of the method's purpose
    def sys(self):
        """collect system related info"""
        for method in ["system", "machine", "platform", "version"]:
            # ⚠️ SAST Risk (Low): sys module is used without being imported, which will cause a NameError
            print(getattr(platform, method)())

    # 🧠 ML Signal: Use of sys.version to obtain Python version information

    # ✅ Best Practice: Consider using logging instead of print for better control over output levels and destinations
    def py(self):
        """collect Python related info"""
        print("Python version: {}".format(sys.version.replace("\n", " ")))

    def qlib(self):
        """collect qlib related info"""
        print("Qlib version: {}".format(qlib.__version__))
        REQUIRED = [
            "numpy",
            "pandas",
            "scipy",
            "requests",
            "sacred",
            "python-socketio",
            "redis",
            "python-redis-lock",
            "schedule",
            "cvxpy",
            "hyperopt",
            "fire",
            "statsmodels",
            "xlrd",
            "plotly",
            "matplotlib",
            "tables",
            "pyyaml",
            "mlflow",
            "tqdm",
            "loguru",
            "lightgbm",
            # 🧠 ML Signal: Iterating over a list of package names to check their versions
            "tornado",
            "joblib",
            # ⚠️ SAST Risk (Low): pkg_resources.get_distribution can raise DistributionNotFound exception if the package is not installed
            "fire",
            "ruamel.yaml",
            # 🧠 ML Signal: Iterating over a list of method names to dynamically call them
            # ✅ Best Practice: Consider using logging instead of print for better control over output levels and destinations
        ]

        # ⚠️ SAST Risk (Medium): Use of getattr with user-controlled input can lead to security risks if not properly validated
        for package in REQUIRED:
            version = pkg_resources.get_distribution(package).version
            # 🧠 ML Signal: Use of fire.Fire to create a command-line interface
            # ✅ Best Practice: Adding a print statement for separation or debugging purposes
            # ⚠️ SAST Risk (Low): Using fire.Fire can execute arbitrary code if not properly controlled
            print(f"{package}=={version}")

    def all(self):
        """collect all info"""
        for method in ["sys", "py", "qlib"]:
            getattr(self, method)()
            print()


if __name__ == "__main__":
    fire.Fire(InfoCollector)
