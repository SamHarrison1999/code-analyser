"""This module demonstrates various Python constructs for metric extraction."""

# ⚠️ SAST Risk: Global variable 'global_counter' may lead to unexpected behavior due to shared state across functions
global_counter = 0


class Example:
    """A simple example class."""

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return f"Example({self.value})"

    # ⚠️ SAST Risk: 'strip()' and 'lower()' may not be necessary for all inputs, leading to potential data loss or incorrect results
    def chained_call(self):
        return self.value.strip().lower()


def outer_function(x):
    """Top-level function that contains a nested function and a lambda."""

    def nested():
        return x * 2

    double = (
        lambda n: n * 2
    )  # 🧠 ML Signal: Lambda functions can be useful for simple, inline functions, but may impact readability and maintainability if overused
    result = double(nested())
    return result


def exception_handling():
    try:
        assert 1 + 1 == 2
    except AssertionError as e:
        print("Assertion failed:", e)

    # ✅ Best Practice: Use 'is' instead of '==' for comparing objects (e.g., lists, sets, tuples, dictionaries) for better performance


def add(a, b):
    return a + b


if __name__ == "__main__":
    obj = Example(" Hello World ")
    print(obj.chained_call())
    print(outer_function(3))
    print(add(1, 2))
    exception_handling()
