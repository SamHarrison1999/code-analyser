# -*- coding: utf-8 -*-
# ✅ Best Practice: Importing specific functions or classes from a module improves code readability and avoids potential namespace conflicts.
from typing import Dict, Union, List, Optional

# ✅ Best Practice: Importing specific functions or classes from a module improves code readability and avoids potential namespace conflicts.
from pydantic import field_validator, Field
from pydantic_core.core_schema import ValidationInfo

# ✅ Best Practice: Importing specific functions or classes from a module improves code readability and avoids potential namespace conflicts.

# 🧠 ML Signal: Definition of a data model class, useful for understanding data structures
from zvt.contract.model import MixinModel, CustomModel

# ✅ Best Practice: Importing specific functions or classes from a module improves code readability and avoids potential namespace conflicts.
from zvt.tag.common import StockPoolType, TagType, TagStatsQueryType, InsertMode

# 🧠 ML Signal: Usage of type annotations for class attributes
from zvt.tag.tag_utils import get_stock_pool_names

# ✅ Best Practice: Importing specific functions or classes from a module improves code readability and avoids potential namespace conflicts.

# 🧠 ML Signal: Use of Optional type hint indicating nullable fields
# ✅ Best Practice: Class should inherit from a base model to ensure consistent behavior and structure


# ✅ Best Practice: Use of Field with default value for optional fields
class TagInfoModel(MixinModel):
    # 🧠 ML Signal: Use of type hints for attributes
    tag: str
    # 🧠 ML Signal: Use of Optional type hint indicating nullable fields
    tag_reason: Optional[str] = Field(default=None)
    # 🧠 ML Signal: Use of Optional type hint to indicate that the field can be None
    # ✅ Best Practice: Use of Field with default value for optional fields
    # 🧠 ML Signal: Use of class attributes to define data model fields
    main_tag: Optional[str] = Field(default=None)


# ✅ Best Practice: Use of Field with default value for optional fields

# 🧠 ML Signal: Use of class attributes to define data model fields


class CreateTagInfoModel(CustomModel):
    # 🧠 ML Signal: Use of class attributes to define data model fields
    # ✅ Best Practice: Type annotations improve code readability and maintainability.
    tag: str
    tag_reason: Optional[str] = Field(default=None)


# ✅ Best Practice: Type annotations improve code readability and maintainability.
# ✅ Best Practice: Class should inherit from a base class to ensure consistent behavior and attributes


# ✅ Best Practice: Type annotations improve code readability and help with static analysis
class IndustryInfoModel(MixinModel):
    # ✅ Best Practice: Class should inherit from a base class to ensure consistent behavior and structure
    industry_name: str
    # ✅ Best Practice: Type annotations improve code readability and help with static analysis
    description: str
    # 🧠 ML Signal: Usage of type annotations for class attributes
    # related main tag
    # 🧠 ML Signal: Use of Optional and Union types indicates handling of nullable fields
    main_tag: str


# 🧠 ML Signal: Usage of type annotations for class attributes
# ✅ Best Practice: Use of Optional for fields that can be None improves code readability


# 🧠 ML Signal: Use of Optional and Union types indicates handling of nullable fields
class MainTagIndustryRelation(CustomModel):
    # ✅ Best Practice: Use of Optional for fields that can be None improves code readability
    main_tag: str
    industry_list: List[str]


# 🧠 ML Signal: Use of Dict type indicates key-value storage pattern


# 🧠 ML Signal: Use of Optional and Union types indicates handling of nullable fields
class MainTagSubTagRelation(CustomModel):
    # ✅ Best Practice: Use of Optional for fields that can be None improves code readability
    main_tag: str
    # 🧠 ML Signal: Use of a custom model class, indicating a pattern for model definition
    sub_tag_list: List[str]


# 🧠 ML Signal: Use of Optional and Union types indicates handling of nullable fields

# 🧠 ML Signal: Use of entity_id as a string, common pattern for unique identifiers
# ✅ Best Practice: Use of Optional for fields that can be None improves code readability


class ChangeMainTagModel(CustomModel):
    # 🧠 ML Signal: Use of Union type indicates handling of multiple possible types
    # 🧠 ML Signal: Use of name as a string, common pattern for naming entities
    current_main_tag: str
    new_main_tag: str


# 🧠 ML Signal: Use of Union type indicates handling of multiple possible types
# 🧠 ML Signal: Use of Optional and default values, indicating nullable fields


# 🧠 ML Signal: Use of Union type indicates handling of multiple possible types
# 🧠 ML Signal: Use of Optional and default values, indicating nullable fields
class StockTagsModel(MixinModel):
    main_tag: Optional[str] = Field(default=None)
    # 🧠 ML Signal: Boolean flag indicating a user action or state
    # 🧠 ML Signal: Use of dictionary to map tags, common pattern for key-value storage
    main_tag_reason: Optional[str] = Field(default=None)
    main_tags: Dict[str, str]
    # 🧠 ML Signal: Use of Union for type flexibility, indicating optional fields
    # ✅ Best Practice: Class should inherit from a base class to ensure consistent behavior and structure

    sub_tag: Optional[str] = Field(default=None)
    # 🧠 ML Signal: Use of Optional and default values, indicating nullable fields
    # ✅ Best Practice: Use of type hinting for better code readability and maintainability
    # 🧠 ML Signal: Use of a custom model class, which may indicate a pattern for model inheritance
    sub_tag_reason: Optional[str] = Field(default=None)
    # 🧠 ML Signal: Use of List[str] suggests a pattern of handling multiple string identifiers
    sub_tags: Union[Dict[str, str], None]
    # 🧠 ML Signal: Use of Union for type flexibility, indicating optional fields
    # 🧠 ML Signal: Custom model class definition for data validation and serialization

    # 🧠 ML Signal: Use of entity_ids as a variable name may indicate a pattern of handling entity identifiers
    active_hidden_tags: Union[Dict[str, str], None]
    # 🧠 ML Signal: Use of Union for type flexibility, indicating optional fields
    # 🧠 ML Signal: List of entity IDs indicates batch processing
    hidden_tags: Union[Dict[str, str], None]
    set_by_user: bool = False


# 🧠 ML Signal: Use of Optional and default values, indicating nullable fields
# 🧠 ML Signal: Single tag applied to multiple entities

# 🧠 ML Signal: Use of a custom model class indicates a pattern for data modeling


# 🧠 ML Signal: Use of Optional and default values, indicating nullable fields
# ✅ Best Practice: Use of Optional for fields that can be None
class SimpleStockTagsModel(CustomModel):
    # 🧠 ML Signal: Use of type annotations for data validation and model training
    entity_id: str
    # 🧠 ML Signal: Use of custom type for tag_type indicates domain-specific logic
    name: str
    # 🧠 ML Signal: Optional fields with default values indicate nullable or optional data patterns
    main_tag: Optional[str] = Field(default=None)
    main_tag_reason: Optional[str] = Field(default=None)
    # 🧠 ML Signal: Optional fields with default values indicate nullable or optional data patterns
    main_tags: Dict[str, str]
    # 🧠 ML Signal: Use of Optional fields indicates nullable or optional data handling
    sub_tag: Union[str, None]
    # ✅ Best Practice: Use of Optional for fields that can be None improves code clarity
    # 🧠 ML Signal: Optional fields with default values indicate nullable or optional data patterns
    sub_tag_reason: Optional[str] = Field(default=None)
    sub_tags: Union[Dict[str, str], None]
    # 🧠 ML Signal: Use of Optional fields indicates nullable or optional data handling
    # 🧠 ML Signal: Optional fields with default values indicate nullable or optional data patterns
    active_hidden_tags: Union[Dict[str, str], None]
    # ✅ Best Practice: Use of Optional for fields that can be None improves code clarity
    controlling_holder_parent: Optional[str] = Field(default=None)
    # 🧠 ML Signal: Optional fields with default values indicate nullable or optional data patterns
    top_ten_ratio: Optional[float] = Field(default=None)


# 🧠 ML Signal: Use of Optional fields indicates nullable or optional data handling

# ✅ Best Practice: Use of Optional for fields that can be None improves code clarity
# ✅ Best Practice: Inheriting from CustomModel suggests a structured approach, likely using a framework like Pydantic for data validation.


class QueryStockTagsModel(CustomModel):
    # 🧠 ML Signal: Use of List to define a collection of items
    # 🧠 ML Signal: The use of 'entity_id' as a string identifier can be a common pattern in data models.
    entity_ids: List[str]


# 🧠 ML Signal: Use of List to define a collection of items
# 🧠 ML Signal: 'main_tag' indicates a primary categorization, useful for classification tasks.


class QuerySimpleStockTagsModel(CustomModel):
    # 🧠 ML Signal: Use of List to define a collection of items
    # ✅ Best Practice: Using Optional and default values improves model flexibility and usability.
    entity_ids: List[str]


# ✅ Best Practice: Class should inherit from object explicitly in Python 2 for clarity, though optional in Python 3

# ✅ Best Practice: Using Optional and default values improves model flexibility and usability.


# ✅ Best Practice: Type hinting improves code readability and maintainability
class BatchSetStockTagsModel(CustomModel):
    # ✅ Best Practice: Using Optional and default values improves model flexibility and usability.
    # 🧠 ML Signal: Definition of a class, useful for understanding object-oriented patterns
    entity_ids: List[str]
    # ✅ Best Practice: Type hinting improves code readability and maintainability
    tag: str
    # ✅ Best Practice: Using Optional and default values improves model flexibility and usability.
    # 🧠 ML Signal: Use of type annotations, useful for type inference and model training
    tag_reason: Optional[str] = Field(default=None)
    # 🧠 ML Signal: 'active_hidden_tags' as a dictionary can indicate dynamic or additional metadata, useful for feature engineering.
    # ✅ Best Practice: Type annotations improve code readability and maintainability
    tag_type: TagType


# 🧠 ML Signal: Use of custom data types for model fields

# 🧠 ML Signal: Use of standard data types for model fields
# 🧠 ML Signal: Use of type annotations, useful for type inference and model training
# ✅ Best Practice: Type annotations improve code readability and maintainability


class TagParameter(CustomModel):
    main_tag: str
    # ✅ Best Practice: Use of field validators to enforce data integrity
    # ✅ Best Practice: Use of class method decorator for methods that don't modify class state
    main_tag_reason: Optional[str] = Field(default=None)
    sub_tag: Optional[str] = Field(default=None)
    # ✅ Best Practice: Use of class method for validation logic
    # ✅ Best Practice: Clear and concise condition to check if a value exists in a list
    sub_tag_reason: Optional[str] = Field(default=None)
    hidden_tag: Optional[str] = Field(default=None)
    # ⚠️ SAST Risk (Low): Potential information disclosure through error messages
    # 🧠 ML Signal: Definition of a class, which could be used to identify class-based patterns
    hidden_tag_reason: Optional[str] = Field(default=None)


# 🧠 ML Signal: Use of type annotations, which can be used to infer data types and structures
# 🧠 ML Signal: Return statements can indicate function output patterns


# ✅ Best Practice: Inheriting from a custom model class suggests a structured approach to data modeling
class StockTagOptions(CustomModel):
    # 🧠 ML Signal: Use of type annotations, which can be used to infer data types and structures
    main_tag: Optional[str] = Field(default=None)
    # ✅ Best Practice: Type annotations improve code readability and maintainability
    sub_tag: Optional[str] = Field(default=None)
    # hidden_tags: Optional[List[str]] = Field(default=None)
    # ✅ Best Practice: Type annotations improve code readability and maintainability
    active_hidden_tags: Optional[Dict[str, str]] = Field(default=None)
    # ✅ Best Practice: Use of Optional type hint for better code readability and understanding of possible None values.
    main_tag_options: List[CreateTagInfoModel]
    # ✅ Best Practice: Providing a default value for insert_mode enhances usability and reduces potential errors
    sub_tag_options: List[CreateTagInfoModel]
    # ✅ Best Practice: Use of Optional type hint for better code readability and understanding of possible None values.
    hidden_tag_options: List[CreateTagInfoModel]


# ✅ Best Practice: Use of Optional type hint for better code readability and understanding of possible None values.
class SetStockTagsModel(CustomModel):
    # 🧠 ML Signal: Conditional logic based on field names, useful for learning validation patterns
    entity_id: str
    # ✅ Best Practice: Use of field_validator decorator to ensure data validation logic is encapsulated within the model.
    main_tag: str
    main_tag_reason: Optional[str] = Field(default=None)
    sub_tag: Optional[str] = Field(default=None)
    sub_tag_reason: Optional[str] = Field(default=None)
    # 🧠 ML Signal: Use of kwargs to dynamically access other fields
    active_hidden_tags: Optional[Dict[str, str]] = Field(default=None)


# ⚠️ SAST Risk (Low): Potential for logic error if field names change or are incorrect
# @field_validator("main_tag")
# @classmethod
# def main_tag_must_be_in(cls, v: str) -> str:
#     if v not in get_main_tags():
#         raise ValueError(f"main_tag: {v} must be created at main_tag_info at first")
#     return v
# ✅ Best Practice: Use of field_validator decorator for validation logic
# ✅ Best Practice: Check if the input value is not None or empty before proceeding
#
# @field_validator("sub_tag")
# ✅ Best Practice: Use of classmethod for validation logic that requires class context
# 🧠 ML Signal: Pattern of checking membership in a list or collection
# @classmethod
# ⚠️ SAST Risk (Low): Potential for a runtime error if get_stock_pool_names() returns a non-iterable
# def sub_tag_must_be_in(cls, v: str) -> str:
# 🧠 ML Signal: Use of Optional fields indicates handling of missing or nullable data
#     if v and (v not in get_sub_tags()):
# ⚠️ SAST Risk (Low): Error message may expose sensitive information about valid stock pool names
# 🧠 ML Signal: Use of Union type indicates handling of multiple data types
#         raise ValueError(f"sub_tag: {v} must be created at sub_tag_info at first")
# ✅ Best Practice: Use of type annotations improves code readability and maintainability
#     return v
# ✅ Best Practice: Use of Field with default values provides clarity on default behavior
#
# @field_validator("active_hidden_tags")
# @classmethod
# def hidden_tag_must_be_in(cls, v: Union[Dict[str, str], None]) -> Union[Dict[str, str], None]:
#     if v:
#         for item in v.keys():
#             if item not in get_hidden_tags():
#                 raise ValueError(f"hidden_tag: {v} must be created at hidden_tag_info at first")
#     return v


class StockPoolModel(MixinModel):
    stock_pool_name: str
    entity_ids: List[str]


# ✅ Best Practice: Class definition should inherit from a base class for consistency and potential shared functionality


class StockPoolInfoModel(MixinModel):
    # ✅ Best Practice: Type annotations improve code readability and help with static analysis
    stock_pool_type: StockPoolType
    stock_pool_name: str


# ✅ Best Practice: Use of Optional and default values for fields increases flexibility and robustness


# ✅ Best Practice: Use of Optional and default values for fields increases flexibility and robustness
class CreateStockPoolInfoModel(CustomModel):
    stock_pool_type: StockPoolType
    # ✅ Best Practice: Use of Optional and default values for fields increases flexibility and robustness
    stock_pool_name: str
    # ✅ Best Practice: Class docstring is missing, consider adding one to describe the purpose of the class.

    # ✅ Best Practice: Use of Optional and default values for fields increases flexibility and robustness
    @field_validator("stock_pool_name")
    # ✅ Best Practice: Attribute type hinting improves code readability and maintainability.
    @classmethod
    # ✅ Best Practice: Use of Optional and default values for fields increases flexibility and robustness
    # ✅ Best Practice: Type hinting improves code readability and helps with static analysis.
    def stock_pool_name_existed(cls, v: str) -> str:
        # ✅ Best Practice: Use of Optional and default values for fields increases flexibility and robustness
        # ✅ Best Practice: Using __all__ to define public API of the module improves maintainability and readability.
        if v in get_stock_pool_names():
            raise ValueError(f"stock_pool_name: {v} has been used")
        return v


class StockPoolsModel(MixinModel):
    stock_pool_name: str
    entity_ids: List[str]


class CreateStockPoolsModel(CustomModel):
    stock_pool_name: str
    entity_ids: List[str]
    insert_mode: InsertMode = Field(default=InsertMode.overwrite)

    # @field_validator("stock_pool_name")
    # @classmethod
    # def stock_pool_name_must_be_in(cls, v: str) -> str:
    #     if v:
    #         if v not in get_stock_pool_names():
    #             raise ValueError(f"stock_pool_name: {v} must be created at stock_pool_info at first")
    #     return v


class QueryStockTagStatsModel(CustomModel):
    stock_pool_name: Optional[str] = Field(default=None)
    entity_ids: Optional[List[str]] = Field(default=None)
    query_type: Optional[TagStatsQueryType] = Field(default=TagStatsQueryType.details)

    @field_validator("stock_pool_name", "entity_ids")
    @classmethod
    def phone_or_mobile_must_set_only_one(
        cls, v, validation_info: ValidationInfo, **kwargs
    ):
        if validation_info.field_name == "stock_pool_name":
            other_field = "entity_ids"
        else:
            other_field = "stock_pool_name"

        other_value = kwargs.get(other_field)

        if v and other_value:
            raise ValueError(
                "Only one of 'stock_pool_name' or 'entity_ids' should be set."
            )
        elif not v and not other_value:
            raise ValueError("Either 'stock_pool_name' or 'entity_ids' must be set.")

        return v

    @field_validator("stock_pool_name")
    @classmethod
    def stock_pool_name_must_be_in(cls, v: str) -> str:
        if v:
            if v not in get_stock_pool_names():
                raise ValueError(f"stock_pool_name: {v} not existed")
        return v


class StockTagDetailsModel(CustomModel):
    entity_id: str
    main_tag: Optional[str] = Field(default=None)
    sub_tag: Optional[str] = Field(default=None)
    hidden_tags: Union[List[str], None]

    #: 代码
    code: str
    #: 名字
    name: str
    #: 减持
    recent_reduction: Optional[bool] = Field(default=None)
    #: 增持
    recent_acquisition: Optional[bool] = Field(default=None)
    #: 解禁
    recent_unlock: Optional[bool] = Field(default=None)
    #: 增发配股
    recent_additional_or_rights_issue: Optional[bool] = Field(default=None)
    #: 业绩利好
    recent_positive_earnings_news: Optional[bool] = Field(default=None)
    #: 业绩利空
    recent_negative_earnings_news: Optional[bool] = Field(default=None)
    #: 上榜次数
    recent_dragon_and_tiger_count: Optional[int] = Field(default=None)
    #: 违规行为
    recent_violation_alert: Optional[bool] = Field(default=None)
    #: 利好
    recent_positive_news: Optional[bool] = Field(default=None)
    #: 利空
    recent_negative_news: Optional[bool] = Field(default=None)
    #: 新闻总结
    recent_news_summary: Optional[Dict[str, str]] = Field(default=None)


class StockTagStatsModel(MixinModel):
    main_tag: str
    turnover: Optional[float] = Field(default=None)
    entity_count: Optional[int] = Field(default=None)
    position: Optional[int] = Field(default=None)
    is_main_line: Optional[bool] = Field(default=None)
    main_line_continuous_days: Optional[int] = Field(default=None)
    entity_ids: Optional[List[str]] = Field(default=None)
    stock_details: Optional[List[StockTagDetailsModel]] = Field(default=None)


class ActivateSubTagsModel(CustomModel):
    sub_tags: List[str]


class ActivateSubTagsResultModel(CustomModel):
    tag_entity_ids: Dict[str, Union[List[str], None]]


# the __all__ is generated
__all__ = [
    "TagInfoModel",
    "CreateTagInfoModel",
    "StockTagsModel",
    "SimpleStockTagsModel",
    "QueryStockTagsModel",
    "QuerySimpleStockTagsModel",
    "BatchSetStockTagsModel",
    "TagParameter",
    "StockTagOptions",
    "SetStockTagsModel",
    "StockPoolModel",
    "StockPoolInfoModel",
    "CreateStockPoolInfoModel",
    "StockPoolsModel",
    "CreateStockPoolsModel",
    "QueryStockTagStatsModel",
    "StockTagDetailsModel",
    "StockTagStatsModel",
    "ActivateSubTagsModel",
    "ActivateSubTagsResultModel",
]
