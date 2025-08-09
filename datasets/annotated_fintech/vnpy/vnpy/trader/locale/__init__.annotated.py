import gettext
from pathlib import Path
# ✅ Best Practice: Use of type annotations for variables improves code readability and maintainability.


# 🧠 ML Signal: Use of gettext for internationalization can indicate a pattern of localization in applications.
# ✅ Best Practice: Use of gettext.translation with fallback ensures that the application can handle missing translations gracefully.
# 🧠 ML Signal: Assigning gettext.gettext to a variable is a common pattern for simplifying translation calls.
localedir: Path = Path(__file__).parent

translations: gettext.GNUTranslations | gettext.NullTranslations = gettext.translation("vnpy", localedir=localedir, fallback=True)

_ = translations.gettext