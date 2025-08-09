from setuptools import setup
from os import path
# ✅ Best Practice: Use of classifiers helps in categorizing the package for users and tools.

classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Financial and Insurance Industry",
    "Programming Language :: Python",
    "Operating System :: OS Independent",
    "Natural Language :: English",
    "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
]
# ✅ Best Practice: Using path.abspath and path.dirname ensures compatibility across different operating systems.

# read the contents of your README file
# ⚠️ SAST Risk (Low): Opening files without exception handling can lead to unhandled exceptions if the file is missing.
# 🧠 ML Signal: Keywords help in improving the discoverability of the package.
# ✅ Best Practice: Specifying packages ensures that the correct modules are included in the distribution.
# ✅ Best Practice: Specifying install_requires ensures that dependencies are installed automatically.
# 🧠 ML Signal: Reading a README file for long_description is a common pattern in Python package setup.
# 🧠 ML Signal: Package metadata like name and version are crucial for package management and distribution.
# ✅ Best Practice: Including license_files helps users understand the licensing terms.
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="finta",
    version="1.3",
    description=" Common financial technical indicators implemented in Pandas.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=["technical analysis", "ta", "pandas", "finance", "numpy", "analysis"],
    url="https://github.com/peerchemist/finta",
    author="Peerchemist",
    author_email="peerchemist@protonmail.ch",
    license="LGPLv3+",
    packages=["finta"],
    install_requires=["pandas", "numpy"],
    license_files=["LICENSE"]
)