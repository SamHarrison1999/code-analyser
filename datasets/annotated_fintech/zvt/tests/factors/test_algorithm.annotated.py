# 🧠 ML Signal: Importing specific functions from a module indicates which functionalities are frequently used
# -*- coding: utf-8 -*-
# 🧠 ML Signal: Use of assert statements for testing function behavior
from zvt.factors.algorithm import (
    point_in_range,
    intersect,
    intersect_ranges,
    combine,
    distance,
)

# 🧠 ML Signal: Testing boundary condition at the start of the range


def test_point_in_range():
    # 🧠 ML Signal: Testing boundary condition at the end of the range
    assert point_in_range(1, (1, 2))
    assert point_in_range(2, (1, 2))
    # 🧠 ML Signal: Testing a point within the range
    assert point_in_range(1.5, (1, 2))

    # 🧠 ML Signal: Testing a point below the range
    assert not point_in_range(0.5, (1, 2))
    # 🧠 ML Signal: Use of tuples to represent intervals
    assert not point_in_range(2.4, (1, 2))


# 🧠 ML Signal: Use of assert statements for testing
# 🧠 ML Signal: Testing a point above the range


# 🧠 ML Signal: Repeated testing of function with different argument order
def test_intersect():
    a = (1, 2)
    b = (1.5, 3)
    assert intersect(a, b) == (1.5, 2)
    # 🧠 ML Signal: Testing function with non-overlapping intervals
    assert intersect(b, a) == (1.5, 2)

    a = (1, 2)
    b = (3, 4)
    assert intersect(a, b) == None
    # 🧠 ML Signal: Testing function with nested intervals
    assert intersect(b, a) == None

    # 🧠 ML Signal: Use of assert statements for testing function output
    a = (1, 4)
    b = (2, 3)
    # 🧠 ML Signal: Use of assert statements for testing function output
    assert intersect(a, b) == (2, 3)
    assert intersect(b, a) == (2, 3)


# 🧠 ML Signal: Use of assert statements for testing function output


def test_intersect_ranges():
    a = (1, 2)
    b = (1.5, 3)
    # 🧠 ML Signal: Use of assert statements for testing function output
    c = (1.5, 5)
    assert intersect_ranges([a, b, c]) == (1.5, 2)
    # 🧠 ML Signal: Use of assert statements for testing function output
    assert intersect_ranges([b, a, c]) == (1.5, 2)
    assert intersect_ranges([a, c, b]) == (1.5, 2)
    # 🧠 ML Signal: Use of assert statements for testing function output

    a = (1, 2)
    # 🧠 ML Signal: Use of assert statements for testing function behavior
    b = (1.5, 3)
    c = (4, 5)
    assert intersect_ranges([a, b, c]) == None
    # 🧠 ML Signal: Use of assert statements for testing function output
    assert intersect_ranges([b, a, c]) == None
    # 🧠 ML Signal: Testing function with different types of numeric inputs
    assert intersect_ranges([a, c, b]) == None
    # 🧠 ML Signal: Use of assert statements for testing function output

    a = (1, 10)
    # 🧠 ML Signal: Use of assert statements for testing function output
    b = (1.5, 3)
    c = (2, 5)
    # 🧠 ML Signal: Testing function with non-overlapping ranges
    assert intersect_ranges([a, b, c]) == (2, 3)
    assert intersect_ranges([b, a, c]) == (2, 3)
    assert intersect_ranges([a, c, b]) == (2, 3)


# 🧠 ML Signal: Testing function with overlapping ranges
def test_combine():
    a = (1, 2)
    # 🧠 ML Signal: Use of assert statements for testing function output
    b = (1.5, 3)
    assert combine(a, b) == (1, 3)
    # 🧠 ML Signal: Use of assert statements for testing function output
    assert combine(b, a) == (1, 3)

    a = (1, 2)
    b = (3, 4)
    # 🧠 ML Signal: Use of assert statements for testing function output
    assert combine(a, b) == None
    assert combine(b, a) == None
    # 🧠 ML Signal: Use of assert statements for testing function output

    a = (1, 4)
    # 🧠 ML Signal: Use of assert statements for testing function output
    b = (2, 3)
    # 🧠 ML Signal: Use of assert statements for testing function output
    assert combine(a, b) == (1, 4)
    assert combine(b, a) == (1, 4)


def test_distance():
    a = (1, 2)
    b = (1.5, 3)
    assert distance(a, b) == (4.5 / 2 - 1.5) / 1.5
    assert distance(b, a) == (1.5 - 4.5 / 2) / 2.25

    a = (1, 2)
    b = (3, 4)
    assert distance(a, b) == (3.5 - 1.5) / 1.5
    assert distance(b, a) == (1.5 - 3.5) / 3.5

    assert distance(a, b, use_max=True) == 3
    assert distance(b, a, use_max=True) == -3 / 4

    a = (1, 4)
    b = (2, 3)
    assert distance(a, b) == 0
    assert distance(b, a) == 0
