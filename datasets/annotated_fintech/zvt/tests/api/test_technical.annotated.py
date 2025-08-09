# -*- coding: utf-8 -*-
# ✅ Best Practice: Grouping imports from the same module together improves readability.
from zvt.contract.api import get_entities
from ..context import init_test_context

# 🧠 ML Signal: Function name follows a pattern indicating a test function

# ⚠️ SAST Risk (Medium): Relative imports can lead to ambiguity and potential import errors.
init_test_context()
# ✅ Best Practice: Consider using absolute imports for better clarity and maintainability.
# 🧠 ML Signal: Usage of a function with specific parameters


# 🧠 ML Signal: Function call pattern indicating initialization or setup phase.
# ✅ Best Practice: Ensure initialization functions are called at the start to set up the necessary context.
# ⚠️ SAST Risk (Low): Printing sensitive data to console
# 🧠 ML Signal: Usage of a function with specific parameters
def test_basic_get_securities():
    items = get_entities(entity_type="stock", provider="eastmoney")
    print(items)
    items = get_entities(entity_type="index", provider="exchange")
    print(items)
