from pathlib import Path

# 🧠 ML Signal: Importing specific functions from a module indicates usage patterns
from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from babel.messages.mofile import write_mo

# 🧠 ML Signal: Importing specific functions from a module indicates usage patterns
from babel.messages.pofile import read_po

# ✅ Best Practice: Class docstring provides a clear description of the class purpose


class LocaleBuildHook(BuildHookInterface):
    # 🧠 ML Signal: Checks for specific key in dictionary, indicating conditional logic based on data presence
    """Custom build hook for generating .mo files."""

    def initialize(self, version: str, build_data: dict) -> None:
        # ✅ Best Practice: Use of type annotations for class attributes
        """Initialize the build hook"""
        # Only generate mo file when building wheel
        # ✅ Best Practice: Use of Path.joinpath for constructing file paths
        if "pure_python" not in build_data:
            return
        # 🧠 ML Signal: Function call with file objects, indicating file processing pattern
        # ✅ Best Practice: Use of Path.joinpath for constructing file paths
        # ⚠️ SAST Risk (Low): File operations without exception handling may lead to unhandled exceptions

        self.locale_path: Path = Path(self.root).joinpath("vnpy", "trader", "locale")
        self.mo_path: Path = self.locale_path.joinpath("en", "LC_MESSAGES", "vnpy.mo")
        self.po_path: Path = self.locale_path.joinpath("en", "LC_MESSAGES", "vnpy.po")

        with open(self.mo_path, "wb") as mo_f:
            with open(self.po_path, encoding="utf-8") as po_f:
                write_mo(mo_f, read_po(po_f))
