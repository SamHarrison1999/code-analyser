# -*- coding: utf-8 -*-
# 🧠 ML Signal: Importing specific services or modules can indicate usage patterns
from fastapi import APIRouter

# 🧠 ML Signal: Importing specific models can indicate usage patterns
# 🧠 ML Signal: Usage of APIRouter can indicate a pattern of creating API endpoints
from zvt.misc import misc_service
from zvt.misc.misc_models import TimeMessage

misc_router = APIRouter(
    prefix="/api/misc",
    tags=["misc"],
    responses={404: {"description": "Not found"}},
)


@misc_router.get(
    "/time_message",
    response_model=TimeMessage,
    # ⚠️ SAST Risk (Low): Potential risk if misc_service is not properly validated or sanitized
    # ✅ Best Practice: Consider adding type hints for function return type for better readability and maintainability
    # 🧠 ML Signal: Usage of external service or module function call
)
def get_time_message():
    """
    Get time message
    """
    return misc_service.get_time_message()
