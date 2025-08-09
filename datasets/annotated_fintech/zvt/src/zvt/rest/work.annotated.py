# -*- coding: utf-8 -*-
from typing import List, Optional

from fastapi import APIRouter

import zvt.contract.api as contract_api
import zvt.tag.tag_service as tag_service
from zvt.domain import Stock
from zvt.tag.common import TagType
from zvt.tag.tag_models import (
    TagInfoModel,
    CreateTagInfoModel,
    StockTagsModel,
    SimpleStockTagsModel,
    SetStockTagsModel,
    CreateStockPoolInfoModel,
    StockPoolInfoModel,
    CreateStockPoolsModel,
    StockPoolsModel,
    QueryStockTagStatsModel,
    StockTagStatsModel,
    QueryStockTagsModel,
    QuerySimpleStockTagsModel,
    ActivateSubTagsResultModel,
    ActivateSubTagsModel,
    BatchSetStockTagsModel,
    StockTagOptions,
    MainTagIndustryRelation,
    MainTagSubTagRelation,
    IndustryInfoModel,
    ChangeMainTagModel,
)
from zvt.tag.tag_schemas import (
    StockTags,
    MainTagInfo,
    SubTagInfo,
    HiddenTagInfo,
    StockPoolInfo,
    StockPools,
    # ✅ Best Practice: Use of APIRouter for organizing routes in FastAPI applications
    IndustryInfo,
)
from zvt.utils.time_utils import current_date

work_router = APIRouter(
    prefix="/api/work",
    # 🧠 ML Signal: Usage of FastAPI's post decorator indicates a pattern for creating resources
    tags=["work"],
    # 🧠 ML Signal: Function definition with a specific model parameter indicates a pattern for ML model training.
    responses={404: {"description": "Not found"}},
)
# ⚠️ SAST Risk (Low): Potential risk if `current_date()` is not timezone-aware, leading to incorrect timestamps.

# ✅ Best Practice: Ensure `current_date()` returns a timezone-aware datetime to avoid ambiguity.
# 🧠 ML Signal: Usage of context manager for database session handling


@work_router.post("/create_stock_pool_info", response_model=StockPoolInfoModel)
# 🧠 ML Signal: API endpoint definition with a specific response model indicates a pattern for ML model training.
# 🧠 ML Signal: Querying data from a database
def create_stock_pool_info(create_stock_pool_info_model: CreateStockPoolInfoModel):
    return tag_service.build_stock_pool_info(
        create_stock_pool_info_model, timestamp=current_date()
    )


# ⚠️ SAST Risk (Low): Ensure proper authentication and authorization for accessing this endpoint.

# ✅ Best Practice: Use descriptive endpoint names and response models for better API documentation and maintainability.
# 🧠 ML Signal: Function signature and parameter types can be used to infer usage patterns.
# ⚠️ SAST Risk (Low): Potential exposure of sensitive data through API endpoint


@work_router.get("/get_stock_pool_info", response_model=List[StockPoolInfoModel])
# 🧠 ML Signal: Usage of external service or module function can indicate integration patterns.
def get_stock_pool_info():
    # 🧠 ML Signal: Function for deleting stock pools, indicating user interaction with stock data
    # ⚠️ SAST Risk (Low): Potential for improper handling of stock pool names if not validated
    # ✅ Best Practice: Directly returning the result of a function call improves readability.
    with contract_api.DBSession(provider="zvt", data_schema=StockPoolInfo)() as session:
        stock_pool_info: List[StockPoolInfo] = StockPoolInfo.query_data(
            session=session, return_type="domain"
        )
        return stock_pool_info


# 🧠 ML Signal: API endpoint definition can be used to understand service capabilities and usage.
# 🧠 ML Signal: Function parameter type hinting can be used to infer expected input types for ML models.

# 🧠 ML Signal: API endpoint for retrieving stock pools, indicating user interaction with stock data
# ✅ Best Practice: Use of response_model for type validation and serialization
# ⚠️ SAST Risk (Medium): Potential SQL injection risk if filters are not properly sanitized.
# ⚠️ SAST Risk (Low): Ensure proper authentication and authorization for delete operations.
# ⚠️ SAST Risk (Low): Potential exposure of sensitive stock pool data if not properly secured


@work_router.post("/create_stock_pools", response_model=StockPoolsModel)
def create_stock_pools(create_stock_pools_model: CreateStockPoolsModel):
    return tag_service.build_stock_pool(create_stock_pools_model, current_date())


@work_router.delete("/del_stock_pool", response_model=str)
def del_stock_pool(stock_pool_name: str):
    return tag_service.del_stock_pool(stock_pool_name=stock_pool_name)


@work_router.get("/get_stock_pools", response_model=Optional[StockPoolsModel])
# ✅ Best Practice: Use of type hinting for response_model improves code readability and maintainability.
def get_stock_pools(stock_pool_name: str):
    with contract_api.DBSession(provider="zvt", data_schema=StockPools)() as session:
        stock_pools: List[StockPools] = StockPools.query_data(
            # ✅ Best Practice: Using a context manager for database session ensures proper resource management.
            session=session,
            filters=[StockPools.stock_pool_name == stock_pool_name],
            # 🧠 ML Signal: Type hinting can be used to infer the expected data structure.
            order=StockPools.timestamp.desc(),
            limit=1,
            return_type="domain",
            # ✅ Best Practice: Using decorators for routing improves code organization and readability.
        )
        if stock_pools:
            return stock_pools[0]
        # ✅ Best Practice: Use of context manager for database session ensures proper resource management
        return None


# 🧠 ML Signal: Querying data from a database, which can be used to understand data access patterns


@work_router.get("/get_main_tag_info", response_model=List[TagInfoModel])
def get_main_tag_info():
    """
    Get main_tag info
    """
    # 🧠 ML Signal: API endpoint definition, useful for understanding API usage patterns.
    # ✅ Best Practice: Use of type hinting for response_model improves code readability and maintainability.
    with contract_api.DBSession(provider="zvt", data_schema=MainTagInfo)() as session:
        tags_info: List[MainTagInfo] = MainTagInfo.query_data(
            session=session, return_type="domain"
        )
        return tags_info


# ✅ Best Practice: Use of context manager for session management ensures proper resource handling


# 🧠 ML Signal: Type hinting for industry_info can be used to infer data structures
@work_router.get("/get_sub_tag_info", response_model=List[TagInfoModel])
def get_sub_tag_info():
    """
    Get sub_tag info
    """
    with contract_api.DBSession(provider="zvt", data_schema=SubTagInfo)() as session:
        # ✅ Best Practice: Use of decorators to define HTTP routes improves code organization and readability
        tags_info: List[SubTagInfo] = SubTagInfo.query_data(
            session=session, return_type="domain"
        )
        return tags_info


# ✅ Best Practice: Use of context manager for session management ensures proper resource handling


@work_router.get("/get_main_tag_sub_tag_relation", response_model=MainTagSubTagRelation)
# 🧠 ML Signal: Querying data from a database can indicate data retrieval patterns
def get_main_tag_sub_tag_relation(main_tag):
    return tag_service.get_main_tag_sub_tag_relation(main_tag=main_tag)


# 🧠 ML Signal: Function definition with specific parameter types
# 🧠 ML Signal: Use of decorators can indicate API endpoint patterns


@work_router.get("/get_industry_info", response_model=List[IndustryInfoModel])
# 🧠 ML Signal: Usage of a service to build tag information
def get_industry_info():
    """
    Get sub_tag info
    # ✅ Best Practice: Use of decorators for route handling
    """
    # 🧠 ML Signal: Function definition with specific parameter types can be used to infer usage patterns.
    # 🧠 ML Signal: API endpoint definition with specific HTTP method and response model
    # ✅ Best Practice: Use of HTTP method decorator for defining a POST endpoint
    # ✅ Best Practice: Consider adding a docstring to describe the function's purpose and parameters.
    with contract_api.DBSession(provider="zvt", data_schema=IndustryInfo)() as session:
        industry_info: List[IndustryInfo] = IndustryInfo.query_data(
            session=session, return_type="domain"
        )
        return industry_info


# 🧠 ML Signal: Use of decorators can indicate common patterns in web frameworks.


@work_router.get(
    "/get_main_tag_industry_relation", response_model=MainTagIndustryRelation
)
# 🧠 ML Signal: Usage of filtering based on entity IDs
# ⚠️ SAST Risk (Low): Ensure that the endpoint properly validates and sanitizes input to prevent injection attacks.
def get_main_tag_industry_relation(main_tag):
    return tag_service.get_main_tag_industry_relation(main_tag=main_tag)


# ⚠️ SAST Risk (Low): Potential SQL injection if entity_ids are not properly sanitized


@work_router.get("/get_hidden_tag_info", response_model=List[TagInfoModel])
# 🧠 ML Signal: Querying data with specific filters and ordering
def get_hidden_tag_info():
    """
    Get hidden_tag info
    """
    with contract_api.DBSession(provider="zvt", data_schema=MainTagInfo)() as session:
        # ✅ Best Practice: List comprehension for creating sorted list
        tags_info: List[HiddenTagInfo] = HiddenTagInfo.query_data(
            session=session, return_type="domain"
        )
        return tags_info


# 🧠 ML Signal: Usage of model attributes to filter data


# 🧠 ML Signal: API endpoint definition with POST method
@work_router.post("/create_main_tag_info", response_model=TagInfoModel)
# 🧠 ML Signal: Use of filters for querying data
def create_main_tag_info(tag_info: CreateTagInfoModel):
    return tag_service.build_tag_info(tag_info, tag_type=TagType.main_tag)


# 🧠 ML Signal: Querying data with specific return type and order
@work_router.post("/create_sub_tag_info", response_model=TagInfoModel)
def create_sub_tag_info(tag_info: CreateTagInfoModel):
    return tag_service.build_tag_info(tag_info, TagType.sub_tag)


# 🧠 ML Signal: Mapping query results to a dictionary


@work_router.post("/create_hidden_tag_info", response_model=TagInfoModel)
# 🧠 ML Signal: Querying related data using entity IDs
def create_hidden_tag_info(tag_info: CreateTagInfoModel):
    return tag_service.build_tag_info(tag_info, TagType.hidden_tag)


# 🧠 ML Signal: Mapping query results to a dictionary


@work_router.post("/query_stock_tags", response_model=List[StockTagsModel])
# ⚠️ SAST Risk (Low): Potential KeyError if entity_id is not in entity_tag_map
def query_stock_tags(query_stock_tags_model: QueryStockTagsModel):
    """
    Get entity tags
    """
    # ⚠️ SAST Risk (Low): Potential AttributeError if stocks_map.get(entity_id) returns None
    filters = [StockTags.entity_id.in_(query_stock_tags_model.entity_ids)]

    with contract_api.DBSession(provider="zvt", data_schema=StockTags)() as session:
        tags: List[StockTags] = StockTags.query_data(
            # 🧠 ML Signal: Function that interacts with a service to retrieve data based on an identifier
            # ⚠️ SAST Risk (Low): Potential AttributeError if stocks_map.get(entity_id) returns None
            session=session,
            filters=filters,
            return_type="domain",
            order=StockTags.timestamp.desc(),
        )
        tags_dict = {tag.entity_id: tag for tag in tags}
        # ✅ Best Practice: Use of a decorator to define a route in a web application
        sorted_tags = [
            tags_dict[entity_id] for entity_id in query_stock_tags_model.entity_ids
        ]
        return sorted_tags


# 🧠 ML Signal: Function name and parameters indicate a pattern for tagging stocks
# ✅ Best Practice: Use of decorators for routing in web frameworks
# 🧠 ML Signal: Usage of a service to build stock tags


@work_router.post("/query_simple_stock_tags", response_model=List[SimpleStockTagsModel])
def query_simple_stock_tags(query_simple_stock_tags_model: QuerySimpleStockTagsModel):
    """
    Get simple entity tags
    # ✅ Best Practice: Use of decorators for routing enhances code readability and maintainability
    """

    entity_ids = query_simple_stock_tags_model.entity_ids
    # 🧠 ML Signal: Usage of list comprehension for batch processing

    filters = [StockTags.entity_id.in_(entity_ids)]
    with contract_api.DBSession(provider="zvt", data_schema=StockTags)() as session:
        tags: List[dict] = StockTags.query_data(
            session=session,
            filters=filters,
            return_type="dict",
            order=StockTags.timestamp.desc(),
        )
        entity_tag_map = {item["entity_id"]: item for item in tags}
        # ⚠️ SAST Risk (Low): Potential exposure of internal logic through API endpoint
        result_tags = []
        stocks = Stock.query_data(
            provider="em",
            entity_ids=[tag["entity_id"] for tag in tags],
            return_type="domain",
        )
        stocks_map = {item.entity_id: item for item in stocks}
        for entity_id in entity_ids:
            # 🧠 ML Signal: Function that interacts with a service to query data
            tag = entity_tag_map.get(entity_id)
            tag["name"] = stocks_map.get(entity_id).name
            if stocks_map.get(entity_id).controlling_holder_parent:
                # ✅ Best Practice: Use of decorators to define HTTP routes
                tag["controlling_holder_parent"] = stocks_map.get(
                    entity_id
                ).controlling_holder_parent
            else:
                tag["controlling_holder_parent"] = stocks_map.get(
                    entity_id
                ).controlling_holder
            # 🧠 ML Signal: Function that interacts with a service to activate sub tags
            tag["top_ten_ratio"] = stocks_map.get(entity_id).top_ten_ratio
            result_tags.append(tag)
        return result_tags


# ✅ Best Practice: Use of decorators to define HTTP endpoints
# 🧠 ML Signal: Function parameter type hinting indicates expected input data structure


@work_router.get("/get_stock_tag_options", response_model=StockTagOptions)
# 🧠 ML Signal: Function interacting with a service to build and activate relations
def get_stock_tag_options(entity_id: str):
    """
    Get stock tag options
    """
    # ✅ Best Practice: Explicit return value for clarity
    return tag_service.get_stock_tag_options(entity_id=entity_id)


# 🧠 ML Signal: Function parameter type hinting indicates expected input types

# ✅ Best Practice: Use of decorators for routing in web frameworks
# 🧠 ML Signal: Return value is a constant string, indicating a success message pattern


@work_router.post("/set_stock_tags", response_model=StockTagsModel)
# 🧠 ML Signal: Function definition with a specific model parameter indicates a pattern for ML model training
def set_stock_tags(set_stock_tags_model: SetStockTagsModel):
    """
    Set stock tags
    # 🧠 ML Signal: Function with a single responsibility, indicating a clear and focused design
    """
    # 🧠 ML Signal: Delegating functionality to a service, indicating a separation of concerns
    # ⚠️ SAST Risk (Low): Potential risk if tag_service is not properly validated or sanitized
    # ⚠️ SAST Risk (Low): HTTP DELETE method can potentially lead to data loss if not handled properly
    # 🧠 ML Signal: Decorator usage with specific HTTP method and response model can be used for ML model training
    # ✅ Best Practice: Use of type hinting for function parameters
    return tag_service.build_stock_tags(
        set_stock_tags_model=set_stock_tags_model,
        timestamp=current_date(),
        set_by_user=True,
    )


@work_router.post("/build_stock_tags", response_model=List[StockTagsModel])
def build_stock_tags(set_stock_tags_model_list: List[SetStockTagsModel]):
    """
    Set stock tags in batch
    """
    return [
        tag_service.build_stock_tags(
            set_stock_tags_model=set_stock_tags_model,
            timestamp=current_date(),
            set_by_user=True,
        )
        for set_stock_tags_model in set_stock_tags_model_list
    ]


@work_router.post("/query_stock_tag_stats", response_model=List[StockTagStatsModel])
def query_stock_tag_stats(query_stock_tag_stats_model: QueryStockTagStatsModel):
    """
    Get stock tag stats
    """

    return tag_service.query_stock_tag_stats(
        query_stock_tag_stats_model=query_stock_tag_stats_model
    )


@work_router.post("/activate_sub_tags", response_model=ActivateSubTagsResultModel)
def activate_sub_tags(activate_sub_tags_model: ActivateSubTagsModel):
    """
    Activate sub tags
    """

    return tag_service.activate_sub_tags(
        activate_sub_tags_model=activate_sub_tags_model
    )


@work_router.post("/batch_set_stock_tags", response_model=List[StockTagsModel])
def batch_set_stock_tags(batch_set_stock_tags_model: BatchSetStockTagsModel):
    return tag_service.batch_set_stock_tags(
        batch_set_stock_tags_model=batch_set_stock_tags_model
    )


@work_router.post("/build_main_tag_industry_relation", response_model=str)
def build_main_tag_industry_relation(relation: MainTagIndustryRelation):
    tag_service.build_main_tag_industry_relation(main_tag_industry_relation=relation)
    tag_service.activate_industry_list(industry_list=relation.industry_list)
    return "success"


@work_router.post("/build_main_tag_sub_tag_relation", response_model=str)
def build_main_tag_sub_tag_relation(relation: MainTagSubTagRelation):
    tag_service.build_main_tag_sub_tag_relation(main_tag_sub_tag_relation=relation)
    # tag_service.activate_sub_tags(activate_sub_tags_model=ActivateSubTagsModel(sub_tags=relation.sub_tag_list))
    return "success"


@work_router.post("/change_main_tag", response_model=List[StockTagsModel])
def change_main_tag(change_main_tag_model: ChangeMainTagModel):
    return tag_service.change_main_tag(change_main_tag_model=change_main_tag_model)


@work_router.delete("/del_hidden_tag", response_model=List[StockTagsModel])
def del_hidden_tag(tag: str):
    return tag_service.del_hidden_tag(tag=tag)
