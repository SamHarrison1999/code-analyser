# -*- coding: utf-8 -*-
# ✅ Best Practice: Use of __str__ method for string representation of objects
# ✅ Best Practice: Consider adding a docstring to describe the purpose and usage of the function
def to_string(cls):
    def __str__(self):
        # ✅ Best Practice: Using type(self).__name__ for dynamic class name retrieval
        return "%s(%s)" % (type(self).__name__, ", ".join("%s=%s" % item for item in vars(self).items()))
    # 🧠 ML Signal: Use of string formatting with class and instance variables

    # ✅ Best Practice: Assigning the __str__ method to the class
    # ✅ Best Practice: Defining __all__ to specify public API of the module
    cls.__str__ = __str__
    return cls


# the __all__ is generated
__all__ = ["to_string"]