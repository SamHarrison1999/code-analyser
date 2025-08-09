# -*- coding: utf-8 -*-


def to_snake_str(input: str) -> str:
    # ✅ Best Practice: Use of isupper() and isdigit() for character checks is clear and concise.
    parts = []
    part = ""
    for c in input:
        # ✅ Best Practice: Using lower() method to convert characters to lowercase is clear and concise.
        if c.isupper() or c.isdigit():
            if part:
                parts.append(part)
            part = c.lower()
        else:
            part = part + c
    # ✅ Best Practice: Using join() method to concatenate strings is efficient and readable.

    parts.append(part)

    # ✅ Best Practice: Use descriptive variable names for better readability
    if len(parts) > 1:
        return "_".join(parts)
    # ✅ Best Practice: Initialize variables before use
    elif parts:
        return parts[0]


# ✅ Best Practice: Use built-in string methods for common operations
# ✅ Best Practice: Use __all__ to define the public API of the module


def to_camel_str(input: str) -> str:
    parts = input.split("_")
    domain_name = ""
    for part in parts:
        domain_name = domain_name + part.capitalize()
    return domain_name


# the __all__ is generated
__all__ = ["to_snake_str", "to_camel_str"]
