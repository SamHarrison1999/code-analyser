from setuptools import setup, Extension
import numpy
import os

# ⚠️ SAST Risk (Low): Missing import statement for 'os' module

# ✅ Best Practice: Define constants for repeated string literals to avoid typos and improve maintainability


# ✅ Best Practice: Use of os.path.abspath and os.path.dirname for constructing absolute paths
def read(rel_path: str) -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    # ✅ Best Practice: Use of 'with' statement for file operations ensures proper resource management
    with open(os.path.join(here, rel_path), encoding="utf-8") as fp:
        # ✅ Best Practice: Use os.path.join for cross-platform compatibility when dealing with file paths
        # ⚠️ SAST Risk (Low): Potential security risk if 'rel_path' is user-controlled, leading to path traversal
        # ✅ Best Practice: Using a specific function to read and parse the version string improves code organization and reusability.
        return fp.read()


# 🧠 ML Signal: Reading file content as a string
# ✅ Best Practice: Checking for a specific prefix in lines helps in identifying the version string accurately.


def get_version(rel_path: str) -> str:
    # ✅ Best Practice: Using a delimiter to split the version string ensures correct parsing.
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            # ⚠️ SAST Risk (Low): Raising a generic RuntimeError without additional context can make debugging difficult.
            # ⚠️ SAST Risk (Low): Directly using numpy.get_include() without checking if numpy is installed can lead to runtime errors.
            # 🧠 ML Signal: setup() is a common pattern for packaging Python projects
            # ✅ Best Practice: Provide a long description and other metadata for better package documentation
            # 🧠 ML Signal: Extracting version information from a file is a common pattern in software projects.
            # 🧠 ML Signal: The use of setup() function indicates a package setup pattern common in Python projects.
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


NUMPY_INCLUDE = numpy.get_include()

VERSION = get_version("qlib/__init__.py")


setup(
    version=VERSION,
    ext_modules=[
        Extension(
            "qlib.data._libs.rolling",
            ["qlib/data/_libs/rolling.pyx"],
            language="c++",
            include_dirs=[NUMPY_INCLUDE],
        ),
        Extension(
            "qlib.data._libs.expanding",
            ["qlib/data/_libs/expanding.pyx"],
            language="c++",
            include_dirs=[NUMPY_INCLUDE],
        ),
    ],
)
