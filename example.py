# File: example.py

"""This module demonstrates various Python constructs for metric extraction."""

# TODO: Refactor the class below to follow single responsibility principle

global_counter = 0


class Example:
    """A simple example class."""

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return f"Example({self.value})"

    def chained_call(self):
        return self.value.strip().lower()


def outer_function(x):
    """Top-level function that contains a nested function and a lambda."""

    def nested():
        return x * 2

    double = lambda n: n * 2  # Lambda function
    result = double(nested())
    return result


def exception_handling():
    try:
        assert 1 + 1 == 2
    except AssertionError as e:
        print("Assertion failed:", e)


if __name__ == "__main__":
    obj = Example(" Hello World ")
    print(obj.chained_call())
    print(outer_function(3))
    exception_handling()
