# -*- coding: utf-8 -*-
# 🧠 ML Signal: Iterating over schema to update model attributes
def update_model(db_model, schema):
    for key, value in schema.dict().items():
        # ⚠️ SAST Risk (Low): Potential for overwriting critical attributes if not validated
        if value is not None:
            # ✅ Best Practice: Explicitly define the public API of the module
            # 🧠 ML Signal: Dynamically setting attributes on an object
            setattr(db_model, key, value)


# the __all__ is generated
__all__ = ["update_model"]
