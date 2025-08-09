import platform
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from fastapi_pagination import Page

import zvt.contract.api as contract_api
import zvt.trading.trading_service as trading_service
from zvt.common.trading_models import BuyParameter, SellParameter, TradingResult
from zvt.trading.trading_models import (
    BuildTradingPlanModel,
    TradingPlanModel,
    QueryTradingPlanModel,
    QueryStockQuoteModel,
    StockQuoteStatsModel,
    QueryStockQuoteSettingModel,
    BuildQueryStockQuoteSettingModel,
    QueryTagQuoteModel,
    TagQuoteStatsModel,
    KdataModel,
    KdataRequestModel,
    TSModel,
    TSRequestModel,
    QuoteStatsModel,
# ✅ Best Practice: Use of APIRouter for organizing routes in FastAPI applications
)
from zvt.trading.trading_schemas import QueryStockQuoteSetting

trading_router = APIRouter(
    prefix="/api/trading",
    tags=["trading"],
    responses={404: {"description": "Not found"}},
# ✅ Best Practice: Use of type hinting for response_model improves code readability and maintainability
# 🧠 ML Signal: Function that interacts with a service to query data
)


# ✅ Best Practice: Use of type hinting for response_model improves code readability and maintainability
# 🧠 ML Signal: Function that interacts with a service, indicating a pattern of service usage.
@trading_router.post("/query_kdata", response_model=Optional[List[KdataModel]])
def query_kdata(kdata_request_model: KdataRequestModel):
    return trading_service.query_kdata(kdata_request_model)
# 🧠 ML Signal: Function returning a service call result, indicating a pattern of service interaction


@trading_router.post("/query_ts", response_model=Optional[List[TSModel]])
# ✅ Best Practice: Use of a decorator to define a route in a web framework
# 🧠 ML Signal: Use of context manager for database session handling
def query_kdata(ts_request_model: TSRequestModel):
    # 🧠 ML Signal: Querying data from a database
    return trading_service.query_ts(ts_request_model)


@trading_router.get("/get_quote_stats", response_model=Optional[QuoteStatsModel])
def get_quote_stats():
    return trading_service.query_quote_stats()


# ⚠️ SAST Risk (Low): Potential exposure of sensitive data through API endpoint
# 🧠 ML Signal: Function uses a service to build a query, indicating a pattern of service delegation
@trading_router.get("/get_query_stock_quote_setting", response_model=Optional[QueryStockQuoteSettingModel])
def get_query_stock_quote_setting():
    with contract_api.DBSession(provider="zvt", data_schema=QueryStockQuoteSetting)() as session:
        # ✅ Best Practice: Use of decorators to define HTTP endpoints improves code organization and readability
        # 🧠 ML Signal: Function uses a service to query data, indicating a pattern of data retrieval.
        query_setting: List[QueryStockQuoteSetting] = QueryStockQuoteSetting.query_data(
            session=session, return_type="domain"
        )
        # ✅ Best Practice: Use of decorators to define HTTP endpoints improves code organization and readability.
        # 🧠 ML Signal: Function signature indicates a pattern of querying stock quotes
        if query_setting:
            return query_setting[0]
        # 🧠 ML Signal: Usage of a service to query stock quotes, indicating a common pattern in trading applications
        return None
# 🧠 ML Signal: Function signature indicates a pattern of using service classes for business logic


# 🧠 ML Signal: Endpoint definition for building a trading plan, useful for identifying API usage patterns
@trading_router.post("/build_query_stock_quote_setting", response_model=QueryStockQuoteSettingModel)
# ✅ Best Practice: Use of decorators for routing improves code organization and readability
# 🧠 ML Signal: Function that interacts with a service to query trading plans
# ✅ Best Practice: Use of decorators for defining API endpoints improves code readability and organization
def build_query_stock_quote_setting(build_query_stock_quote_setting_model: BuildQueryStockQuoteSettingModel):
    return trading_service.build_query_stock_quote_setting(build_query_stock_quote_setting_model)
# 🧠 ML Signal: Function definition for retrieving current trading plan

# ✅ Best Practice: Use of decorators to define HTTP endpoints
# 🧠 ML Signal: Usage of trading_service to get trading plan

@trading_router.post("/query_tag_quotes", response_model=List[TagQuoteStatsModel])
def query_tag_quotes(query_tag_quote_model: QueryTagQuoteModel):
    # ✅ Best Practice: Use of decorator for defining a route in a web application
    # 🧠 ML Signal: Function that retrieves trading plans, indicating user interest in trading activities
    return trading_service.query_tag_quotes(query_tag_quote_model)


# ✅ Best Practice: Use of decorators to define HTTP endpoints in a web framework
# 🧠 ML Signal: Conditional import based on platform, indicating platform-specific behavior
@trading_router.post("/query_stock_quotes", response_model=Optional[StockQuoteStatsModel])
def query_stock_quotes(query_stock_quote_model: QueryStockQuoteModel):
    # ⚠️ SAST Risk (Medium): Dynamic import based on condition, could lead to import errors or security issues
    return trading_service.query_stock_quotes(query_stock_quote_model)

# 🧠 ML Signal: Platform-specific method call, indicating usage pattern for Windows

@trading_router.post("/build_trading_plan", response_model=TradingPlanModel)
# 🧠 ML Signal: Function definition with specific parameter type hints
def build_trading_plan(build_trading_plan_model: BuildTradingPlanModel):
    # ⚠️ SAST Risk (Low): Generic exception message, could be more informative
    return trading_service.build_trading_plan(build_trading_plan_model)
# 🧠 ML Signal: Checking for platform-specific conditions

# ✅ Best Practice: Decorator usage for route definition, improves code organization and readability
# ⚠️ SAST Risk (Low): Dynamic import based on platform condition

@trading_router.post("/query_trading_plan", response_model=Page[TradingPlanModel])
# 🧠 ML Signal: Platform-specific method call
# ⚠️ SAST Risk (Low): Generic exception handling with HTTP status code
def query_trading_plan(query_trading_plan_model: QueryTradingPlanModel):
    return trading_service.query_trading_plan(query_trading_plan_model)


@trading_router.get("/get_current_trading_plan", response_model=List[TradingPlanModel])
def get_current_trading_plan():
    return trading_service.get_current_trading_plan()


@trading_router.get("/get_future_trading_plan", response_model=List[TradingPlanModel])
def get_future_trading_plan():
    return trading_service.get_future_trading_plan()


@trading_router.post("/buy", response_model=TradingResult)
def buy(buy_position_strategy: BuyParameter):
    if platform.system() == "Windows":
        from zvt.broker.qmt.context import qmt_context

        return qmt_context.qmt_account.buy(buy_position_strategy)
    else:
        raise HTTPException(status_code=500, detail="Please use qmt in windows! ")


@trading_router.post("/sell", response_model=TradingResult)
def sell(sell_position_strategy: SellParameter):
    if platform.system() == "Windows":
        from zvt.broker.qmt.context import qmt_context

        return qmt_context.qmt_account.sell(sell_position_strategy)
    else:
        raise HTTPException(status_code=500, detail="Please use qmt in windows! ")