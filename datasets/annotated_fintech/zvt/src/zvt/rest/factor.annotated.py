# -*- coding: utf-8 -*-
from typing import List
# ✅ Best Practice: Grouping imports by standard, third-party, and local modules improves readability.

from fastapi import APIRouter

# ✅ Best Practice: Using APIRouter for route management in FastAPI improves modularity and maintainability.
from zvt.contract import zvt_context
from zvt.factors import factor_service
from zvt.factors.factor_models import FactorRequestModel, TradingSignalModel

factor_router = APIRouter(
    prefix="/api/factor",
    tags=["factor"],
    # 🧠 ML Signal: Usage of list comprehension to iterate over dictionary keys
    responses={404: {"description": "Not found"}},
)
# ✅ Best Practice: Using type hints for response_model improves code readability and helps with static analysis.
# 🧠 ML Signal: Function definition with a specific parameter type can indicate usage patterns

# ✅ Best Practice: Use of FastAPI's decorator for defining a POST endpoint
# ✅ Best Practice: Function name is descriptive and follows naming conventions
# ✅ Best Practice: Single responsibility function, which enhances maintainability
# ⚠️ SAST Risk (Low): Directly passing user input to a service call without validation or sanitization

@factor_router.get("/get_factors", response_model=List[str])
def get_factors():
    return [name for name in zvt_context.factor_cls_registry.keys()]


@factor_router.post("/query_factor_result", response_model=List[TradingSignalModel])
def query_factor_result(factor_request_model: FactorRequestModel):
    return factor_service.query_factor_result(factor_request_model)