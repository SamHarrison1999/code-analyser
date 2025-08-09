# -*- coding: utf-8 -*-
# 🧠 ML Signal: Importing specific functions from a module indicates usage patterns and dependencies
from zvt.contract.api import get_entities
from zvt.utils.utils import iterate_with_step, to_str, float_to_pct

# 🧠 ML Signal: Importing multiple utilities from a module suggests common utility functions used in the codebase


def test_iterate_with_step():
    # 🧠 ML Signal: Iterating over data in chunks or steps
    data = range(1000)
    first = None
    last = None
    for sub_data in iterate_with_step(data):
        if not first:
            # ✅ Best Practice: Use of assertions for testing expected outcomes
            first = sub_data
        last = sub_data

    # 🧠 ML Signal: Function definition for testing, indicating a test pattern
    assert first[0] == 0
    assert first[-1] == 99
    # 🧠 ML Signal: Function call with specific parameter, indicating usage pattern

    assert last[0] == 900
    # ✅ Best Practice: Initialize variables before use
    assert last[-1] == 999


# ✅ Best Practice: Initialize variables before use


def test_iterate_entities():
    # 🧠 ML Signal: Iteration pattern over a custom iterator
    data = get_entities(entity_type="stock")
    first = None
    # ✅ Best Practice: Check for None before assignment
    # 🧠 ML Signal: Function testing pattern
    last = None
    for sub_data in iterate_with_step(data):
        # 🧠 ML Signal: Testing for None input
        if first is None:
            first = sub_data
        # ⚠️ SAST Risk (Low): Assumes 'first' is not None and has a length of 100
        # 🧠 ML Signal: Testing for empty string input
        last = sub_data

    # 🧠 ML Signal: Use of assert statements for testing function output
    # ⚠️ SAST Risk (Low): Assumes 'last' is not None and has a length <= 100
    # 🧠 ML Signal: Testing for single character string input
    assert len(first) == 100
    assert len(last) <= 100


# 🧠 ML Signal: Testing conversion of float to percentage string
# 🧠 ML Signal: Testing for list of strings input


# 🧠 ML Signal: Testing conversion of float to percentage string
# 🧠 ML Signal: Testing for list of integers input
def test_to_str():
    # 🧠 ML Signal: Testing conversion of float to percentage string
    assert to_str(None) is None
    assert to_str("") is None
    assert to_str("a") == "a"
    assert to_str(["a", "b"]) == "a;b"
    assert to_str([1, 2]) == "1;2"


def test_float_to_pct():
    assert float_to_pct(0.1) == "10.00%"
    assert float_to_pct(0.111) == "11.10%"
    assert float_to_pct(0.8) == "80.00%"
    assert float_to_pct(0.555) == "55.50%"
    assert float_to_pct(0.33333) == "33.33%"
