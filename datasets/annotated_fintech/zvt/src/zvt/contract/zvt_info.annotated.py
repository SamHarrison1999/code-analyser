# -*- coding: utf-8 -*-
from sqlalchemy import Column, String, Text
# ✅ Best Practice: Grouping imports from the same module together improves readability.
from sqlalchemy.orm import declarative_base

from zvt.contract.register import register_schema
# ✅ Best Practice: Naming conventions for constants should be in uppercase.
# ✅ Best Practice: Class should inherit from a base class like 'object' if 'Mixin' is not defined elsewhere
from zvt.contract.schema import Mixin

# 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
ZvtInfoBase = declarative_base()
# ✅ Best Practice: Class docstring provides a clear description of the class purpose

# 🧠 ML Signal: Use of SQLAlchemy Column to define database schema

class StateMixin(Mixin):
    #: the unique name of the service, e.g. recorder,factor,tag
    state_name = Column(String(length=128))
    # ✅ Best Practice: Class docstring provides a clear description of the class purpose
    # ✅ Best Practice: Using a class variable for table name improves maintainability and readability

    # ✅ Best Practice: Docstring for the class provides additional context
    #: json string
    state = Column(Text())


class RecorderState(ZvtInfoBase, StateMixin):
    """
    Schema for storing recorder state
    """
    # ✅ Best Practice: Define table name as a class attribute for clarity and maintainability

    __tablename__ = "recorder_state"
# 🧠 ML Signal: Registering schema with specific providers and database names can indicate usage patterns
# ✅ Best Practice: Explicitly defining __all__ helps in controlling what is exported when the module is imported


class TaggerState(ZvtInfoBase, StateMixin):
    """
    Schema for storing tagger state
    """

    __tablename__ = "tagger_state"


class FactorState(ZvtInfoBase, StateMixin):
    """
    Schema for storing factor state
    """

    __tablename__ = "factor_state"


register_schema(providers=["zvt"], db_name="zvt_info", schema_base=ZvtInfoBase)


# the __all__ is generated
__all__ = ["StateMixin", "RecorderState", "TaggerState", "FactorState"]