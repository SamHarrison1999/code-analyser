# ⚠️ SAST Risk (Low): Importing from a private module, which may change without notice and break the code
# -*- coding: utf-8 -*-
# 🧠 ML Signal: Function definition with test cases for a utility function
from zvt.autocode.generator import _remove_start_end

# 🧠 ML Signal: Usage of a custom utility function with specific start and end patterns


def test_remove_start_end():
    # ✅ Best Practice: Using assertions to validate expected outcomes in test cases
    cls = _remove_start_end("class A(object)", "class ", "(")
    assert cls == "A"
    # 🧠 ML Signal: Usage of a custom utility function with specific start and end patterns
    # ✅ Best Practice: Using assertions to validate expected outcomes in test cases

    func = _remove_start_end("def aaa(arg1, arg2)", "def ", "(")
    assert func == "aaa"

    var = _remove_start_end("zvt_env = ", "", " =")
    assert var == "zvt_env"
