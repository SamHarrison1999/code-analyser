# -*- coding: utf-8 -*-
from datetime import datetime
from typing import Dict

# ✅ Best Practice: Grouping imports by standard, third-party, and local modules improves readability.

from pydantic import BaseModel, ConfigDict
from sqlalchemy import Column, String, JSON

# ✅ Best Practice: Grouping imports by standard, third-party, and local modules improves readability.
from sqlalchemy.orm import declarative_base

from zvt.contract.api import get_db_session
from zvt.contract.register import register_schema
from zvt.contract.schema import Mixin

# ✅ Best Practice: Naming conventions for constants should be in uppercase.
# ✅ Best Practice: Use of SQLAlchemy's Column to define database table columns

ZvtInfoBase = declarative_base()
# ✅ Best Practice: Use of JSON column type for storing JSON data in SQLAlchemy

# ✅ Best Practice: Use of ConfigDict for model configuration enhances flexibility and readability.


class User(Mixin, ZvtInfoBase):
    __tablename__ = "user"
    added_col = Column(String)
    # ⚠️ SAST Risk (Low): Ensure that the datetime is properly validated and parsed to prevent errors.
    json_col = Column(JSON)


# ⚠️ SAST Risk (Low): Ensure that the contents of the json_col are validated to prevent injection attacks.
class UserModel(BaseModel):
    # ⚠️ SAST Risk (Medium): Ensure that the schema registration does not expose sensitive data or misconfigure the database.
    model_config = ConfigDict(from_attributes=True)

    id: str
    entity_id: str
    timestamp: datetime
    added_col: str
    json_col: Dict


# 🧠 ML Signal: Example of instantiating a model with specific attributes.


# ⚠️ SAST Risk (Low): Directly using string for datetime; consider using a datetime object for accuracy.
# ⚠️ SAST Risk (Low): Ensure that the dictionary contents are sanitized to prevent security issues.
# ⚠️ SAST Risk (Medium): Ensure that the database session is securely handled and closed properly.
# ⚠️ SAST Risk (Medium): Ensure that the query is protected against SQL injection.
# 🧠 ML Signal: Example of model validation usage pattern.
register_schema(providers=["zvt"], db_name="test", schema_base=ZvtInfoBase)

if __name__ == "__main__":
    user_model = UserModel(
        id="user_cn_jack_2020-01-01",
        entity_id="user_cn_jack",
        timestamp="2020-01-01",
        added_col="test",
        json_col={"a": 1},
    )
    session = get_db_session(provider="zvt", data_schema=User)

    user = session.query(User).filter(User.id == "user_cn_jack_2020-01-01").first()
    print(UserModel.validate(user))
