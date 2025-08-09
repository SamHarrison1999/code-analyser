# -*- coding: utf-8 -*-
import logging

# ✅ Best Practice: Grouping imports into standard library, third-party, and local application sections improves readability.
from typing import Type

from zvt.contract import Mixin
from zvt.contract import TradableEntity
from zvt.contract.api import get_db_session
from zvt.contract.base_service import OneStateService
from zvt.contract.zvt_info import TaggerState
from zvt.domain import Stock

# ✅ Best Practice: Class should inherit from a base class to ensure consistent behavior and structure
# ✅ Best Practice: Using a logger with __name__ ensures that the log messages are correctly associated with the module name.
from zvt.tag.tag_schemas import StockTags

# ✅ Best Practice: Defining state_schema for the class to ensure consistent state management
logger = logging.getLogger(__name__)

# ✅ Best Practice: Type hinting for entity_schema to improve code readability and maintainability


class Tagger(OneStateService):
    # ✅ Best Practice: Type hinting for data_schema to improve code readability and maintainability
    state_schema = TaggerState
    # ⚠️ SAST Risk (Low): Use of assert for runtime checks can be disabled with optimization flags

    # 🧠 ML Signal: Hardcoded start timestamp could be used to identify patterns or trends over time
    entity_schema: Type[TradableEntity] = None
    # ⚠️ SAST Risk (Low): Use of assert for runtime checks can be disabled with optimization flags

    data_schema: Type[Mixin] = None

    # 🧠 ML Signal: Usage of a specific database session provider
    start_timestamp = "2018-01-01"

    def __init__(self, force=False) -> None:
        # ✅ Best Practice: Method should have a docstring explaining its purpose
        # 🧠 ML Signal: Logging usage pattern for state management
        super().__init__()
        assert self.entity_schema is not None
        # ✅ Best Practice: Consider providing a custom error message for clarity
        # ✅ Best Practice: Class definition should inherit from a base class to promote code reuse and maintainability
        assert self.data_schema is not None
        # 🧠 ML Signal: Logging usage pattern for tracking execution flow
        self.force = force
        # ✅ Best Practice: Class attributes should be defined at the top of the class for better readability
        self.session = get_db_session(provider="zvt", data_schema=self.data_schema)
        # ✅ Best Practice: Method should be defined within a class
        if self.state and not self.force:
            # ✅ Best Practice: Class attributes should be defined at the top of the class for better readability
            logger.info("get start_timestamp from state")
            # ✅ Best Practice: __all__ is used to define the public interface of the module
            # ✅ Best Practice: Raising NotImplementedError is a common pattern for abstract methods
            self.start_timestamp = self.state["current_timestamp"]
        logger.info(f"tag start_timestamp: {self.start_timestamp}")

    def tag(self):
        raise NotImplementedError


class StockTagger(Tagger):
    data_schema = StockTags
    entity_schema = Stock

    def tag(self):
        raise NotImplementedError


# the __all__ is generated
__all__ = ["Tagger", "StockTagger"]
