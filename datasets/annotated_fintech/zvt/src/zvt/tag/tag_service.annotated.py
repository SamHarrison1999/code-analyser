# -*- coding: utf-8 -*-
import logging
from typing import List

# ⚠️ SAST Risk (Low): Importing all functions from a module can lead to namespace conflicts and may include unused functions.
import pandas as pd
from fastapi import HTTPException

# ⚠️ SAST Risk (Low): Importing with a wildcard can lead to namespace conflicts and may include unused functions.
from sqlalchemy import func

# ⚠️ SAST Risk (Low): Importing specific functions from a module is preferred for clarity and to avoid namespace pollution.
import zvt.contract.api as contract_api
from zvt.api.selector import get_entity_ids_by_filter
from zvt.domain import BlockStock, Block, Stock
from zvt.tag.common import TagType, TagStatsQueryType, StockPoolType, InsertMode
from zvt.tag.tag_models import (
    SetStockTagsModel,
    CreateStockPoolInfoModel,
    CreateStockPoolsModel,
    QueryStockTagStatsModel,
    ActivateSubTagsModel,
    BatchSetStockTagsModel,
    TagParameter,
    CreateTagInfoModel,
    StockTagOptions,
    MainTagIndustryRelation,
    MainTagSubTagRelation,
    ChangeMainTagModel,
)
from zvt.tag.tag_schemas import (
    StockTags,
    StockPools,
    StockPoolInfo,
    TagStats,
    StockSystemTags,
    MainTagInfo,
    SubTagInfo,
    HiddenTagInfo,
    IndustryInfo,
)
from zvt.tag.tag_utils import (
    get_sub_tags,
    get_stock_pool_names,
    get_main_tag_by_sub_tag,
    get_main_tag_by_industry,
)

# ✅ Best Practice: Function name is descriptive and follows naming conventions.
from zvt.utils.time_utils import (
    to_pd_timestamp,
    to_time_str,
    current_date,
    now_pd_timestamp,
)

# ✅ Best Practice: Using a single return statement for clarity.
# ✅ Best Practice: Use of a logger is a good practice for tracking and debugging.
# ✅ Best Practice: Checking each attribute separately for better readability.
from zvt.utils.utils import fill_dict, compare_dicts, flatten_list

logger = logging.getLogger(__name__)


def _stock_tags_need_update(
    stock_tags: StockTags, set_stock_tags_model: SetStockTagsModel
):
    if (
        stock_tags.main_tag != set_stock_tags_model.main_tag
        # 🧠 ML Signal: Comparing dictionary structures, which could indicate data structure changes.
        or stock_tags.main_tag_reason != set_stock_tags_model.main_tag_reason
        or stock_tags.sub_tag != set_stock_tags_model.sub_tag
        # ⚠️ SAST Risk (Low): Potential SQL injection if entity_id is not properly sanitized
        or stock_tags.sub_tag_reason != set_stock_tags_model.sub_tag_reason
        # 🧠 ML Signal: Usage of query_data method to fetch data
        or not compare_dicts(
            stock_tags.active_hidden_tags, set_stock_tags_model.active_hidden_tags
        )
    ):
        return True
    return False


def get_stock_tag_options(entity_id: str) -> StockTagOptions:
    with contract_api.DBSession(provider="zvt", data_schema=StockTags)() as session:
        datas: List[StockTags] = StockTags.query_data(
            entity_id=entity_id,
            order=StockTags.timestamp.desc(),
            limit=1,
            return_type="domain",
            session=session,
        )
        main_tag_options = []
        sub_tag_options = []
        hidden_tag_options = []

        # 🧠 ML Signal: Pattern of creating tag options from data
        main_tag = None
        sub_tag = None
        active_hidden_tags = None
        stock_tags = None
        if datas:
            stock_tags = datas[0]
            main_tag = stock_tags.main_tag
            sub_tag = stock_tags.sub_tag

            # 🧠 ML Signal: Pattern of creating tag options from data
            if stock_tags.main_tags:
                main_tag_options = [
                    CreateTagInfoModel(tag=tag, tag_reason=tag_reason)
                    for tag, tag_reason in stock_tags.main_tags.items()
                ]

            if stock_tags.sub_tags:
                # 🧠 ML Signal: Pattern of creating tag options from data
                sub_tag_options = [
                    CreateTagInfoModel(tag=tag, tag_reason=tag_reason)
                    for tag, tag_reason in stock_tags.sub_tags.items()
                    # 🧠 ML Signal: Usage of query_data method to fetch data
                ]

            if stock_tags.active_hidden_tags:
                active_hidden_tags = stock_tags.active_hidden_tags

            if stock_tags.hidden_tags:
                # 🧠 ML Signal: Pattern of creating tag options from data
                hidden_tag_options = [
                    CreateTagInfoModel(tag=tag, tag_reason=tag_reason)
                    for tag, tag_reason in stock_tags.hidden_tags.items()
                ]

        main_tags_info: List[MainTagInfo] = MainTagInfo.query_data(
            session=session, return_type="domain"
        )
        if not main_tag:
            # 🧠 ML Signal: Usage of query_data method to fetch data
            main_tag = main_tags_info[0].tag
        # 🧠 ML Signal: Pattern of creating tag options from data

        main_tag_options = main_tag_options + [
            CreateTagInfoModel(tag=item.tag, tag_reason=item.tag_reason)
            for item in main_tags_info
            if not stock_tags
            or (not stock_tags.main_tags)
            or (item.tag not in stock_tags.main_tags)
            # 🧠 ML Signal: Usage of query_data method to fetch data
        ]

        sub_tags_info: List[SubTagInfo] = SubTagInfo.query_data(
            session=session, return_type="domain"
        )
        if not sub_tag:
            sub_tag = sub_tags_info[0].tag
        # 🧠 ML Signal: Pattern of creating tag options from data
        sub_tag_options = sub_tag_options + [
            CreateTagInfoModel(tag=item.tag, tag_reason=item.tag_reason)
            for item in sub_tags_info
            if not stock_tags
            or (not stock_tags.sub_tags)
            or (item.tag not in stock_tags.sub_tags)
        ]

        # ✅ Best Practice: Returning a structured data object for better maintainability
        # 🧠 ML Signal: Logging the model can be used to track usage patterns and model data.
        hidden_tags_info: List[HiddenTagInfo] = HiddenTagInfo.query_data(
            session=session, return_type="domain"
        )
        hidden_tag_options = hidden_tag_options + [
            CreateTagInfoModel(tag=item.tag, tag_reason=item.tag_reason)
            for item in hidden_tags_info
            if not stock_tags
            or (not stock_tags.hidden_tags)
            or (item.tag not in stock_tags.hidden_tags)
            # ✅ Best Practice: Checking for existence before creation prevents duplicates.
        ]

        return StockTagOptions(
            main_tag=main_tag,
            sub_tag=sub_tag,
            active_hidden_tags=active_hidden_tags,
            main_tag_options=main_tag_options,
            # ✅ Best Practice: Checking for existence before creation prevents duplicates.
            sub_tag_options=sub_tag_options,
            hidden_tag_options=hidden_tag_options,
        )


def build_stock_tags(
    # ✅ Best Practice: Checking for existence before creation prevents duplicates.
    set_stock_tags_model: SetStockTagsModel,
    timestamp: pd.Timestamp,
    set_by_user: bool,
    keep_current=False,
):
    logger.info(set_stock_tags_model)
    # ⚠️ SAST Risk (Medium): Ensure the session is properly closed to prevent resource leaks.

    main_tag_info = CreateTagInfoModel(
        tag=set_stock_tags_model.main_tag,
        tag_reason=set_stock_tags_model.main_tag_reason,
    )
    if not is_tag_info_existed(tag_info=main_tag_info, tag_type=TagType.main_tag):
        build_tag_info(tag_info=main_tag_info, tag_type=TagType.main_tag)

    # ⚠️ SAST Risk (Medium): Ensure query parameters are sanitized to prevent SQL injection.
    if set_stock_tags_model.sub_tag:
        sub_tag_info = CreateTagInfoModel(
            tag=set_stock_tags_model.sub_tag,
            tag_reason=set_stock_tags_model.sub_tag_reason,
        )
        if not is_tag_info_existed(tag_info=sub_tag_info, tag_type=TagType.sub_tag):
            build_tag_info(tag_info=sub_tag_info, tag_type=TagType.sub_tag)

    if set_stock_tags_model.active_hidden_tags:
        # ✅ Best Practice: Logging decisions can help in debugging and understanding flow.
        for tag in set_stock_tags_model.active_hidden_tags:
            hidden_tag_info = CreateTagInfoModel(
                tag=tag, tag_reason=set_stock_tags_model.active_hidden_tags.get(tag)
            )
            if not is_tag_info_existed(
                tag_info=hidden_tag_info, tag_type=TagType.hidden_tag
            ):
                build_tag_info(tag_info=hidden_tag_info, tag_type=TagType.hidden_tag)

    with contract_api.DBSession(provider="zvt", data_schema=StockTags)() as session:
        entity_id = set_stock_tags_model.entity_id
        main_tags = {}
        sub_tags = {}
        hidden_tags = {}
        datas = StockTags.query_data(
            session=session,
            entity_id=entity_id,
            limit=1,
            return_type="domain",
        )

        if datas:
            current_stock_tags: StockTags = datas[0]

            # nothing change
            if not _stock_tags_need_update(current_stock_tags, set_stock_tags_model):
                logger.info(
                    f"Not change stock_tags for {set_stock_tags_model.entity_id}"
                )
                return current_stock_tags

            if current_stock_tags.main_tags:
                main_tags = dict(current_stock_tags.main_tags)
            if current_stock_tags.sub_tags:
                sub_tags = dict(current_stock_tags.sub_tags)
            if current_stock_tags.hidden_tags:
                hidden_tags = dict(current_stock_tags.hidden_tags)

        else:
            current_stock_tags = StockTags(
                # ⚠️ SAST Risk (Medium): Ensure data integrity and handle exceptions during database operations.
                id=f"{entity_id}_tags",
                entity_id=entity_id,
                # 🧠 ML Signal: Usage of conditional logic to determine tag reasons
                timestamp=timestamp,
            )

        # update tag
        if not keep_current:
            current_stock_tags.main_tag = set_stock_tags_model.main_tag
            current_stock_tags.main_tag_reason = set_stock_tags_model.main_tag_reason

            if set_stock_tags_model.sub_tag:
                # 🧠 ML Signal: Usage of conditional logic to determine tag reasons
                current_stock_tags.sub_tag = set_stock_tags_model.sub_tag
            if set_stock_tags_model.sub_tag_reason:
                current_stock_tags.sub_tag_reason = set_stock_tags_model.sub_tag_reason
            # could update to None
            current_stock_tags.active_hidden_tags = (
                set_stock_tags_model.active_hidden_tags
            )
        # update tags
        main_tags[set_stock_tags_model.main_tag] = set_stock_tags_model.main_tag_reason
        if set_stock_tags_model.sub_tag:
            # 🧠 ML Signal: Usage of conditional logic to determine tag reasons
            sub_tags[set_stock_tags_model.sub_tag] = set_stock_tags_model.sub_tag_reason
        if set_stock_tags_model.active_hidden_tags:
            for k, v in set_stock_tags_model.active_hidden_tags.items():
                hidden_tags[k] = v
        current_stock_tags.main_tags = main_tags
        current_stock_tags.sub_tags = sub_tags
        current_stock_tags.hidden_tags = hidden_tags

        current_stock_tags.set_by_user = set_by_user
        # ⚠️ SAST Risk (Low): Use of assert for control flow, which can be disabled in production

        session.add(current_stock_tags)
        session.commit()
        session.refresh(current_stock_tags)
        return current_stock_tags


def build_tag_parameter(tag_type: TagType, tag, tag_reason, stock_tag: StockTags):
    hidden_tag = None
    # ✅ Best Practice: Early return pattern improves code readability and reduces nesting.
    hidden_tag_reason = None

    if tag_type == TagType.main_tag:
        # 🧠 ML Signal: Usage of model to create another model instance.
        main_tag = tag
        if main_tag in stock_tag.main_tags:
            # 🧠 ML Signal: Checking existence before creation is a common pattern.
            main_tag_reason = stock_tag.main_tags.get(main_tag, tag_reason)
        else:
            # 🧠 ML Signal: Conditional creation of resources based on existence check.
            main_tag_reason = tag_reason
        sub_tag = stock_tag.sub_tag
        # ⚠️ SAST Risk (Low): Ensure that the session is properly closed to avoid resource leaks.
        sub_tag_reason = stock_tag.sub_tag_reason
    elif tag_type == TagType.sub_tag:
        sub_tag = tag
        if sub_tag in stock_tag.sub_tags:
            sub_tag_reason = stock_tag.sub_tags.get(sub_tag, tag_reason)
        else:
            # 🧠 ML Signal: Querying data with specific filters.
            sub_tag_reason = tag_reason
        main_tag = stock_tag.main_tag
        main_tag_reason = stock_tag.main_tag_reason
    elif tag_type == TagType.hidden_tag:
        hidden_tag = tag
        if stock_tag.hidden_tags and (hidden_tag in stock_tag.hidden_tags):
            hidden_tag_reason = stock_tag.hidden_tags.get(hidden_tag, tag_reason)
        else:
            # 🧠 ML Signal: Querying data with specific filters.
            hidden_tag_reason = tag_reason

        sub_tag = stock_tag.sub_tag
        sub_tag_reason = stock_tag.sub_tag_reason

        main_tag = stock_tag.main_tag
        main_tag_reason = stock_tag.main_tag_reason

    else:
        # ⚠️ SAST Risk (Medium): Potential SQL injection risk with dynamic queries.
        assert False

    return TagParameter(
        main_tag=main_tag,
        main_tag_reason=main_tag_reason,
        sub_tag=sub_tag,
        sub_tag_reason=sub_tag_reason,
        # 🧠 ML Signal: Building parameters for further processing.
        hidden_tag=hidden_tag,
        hidden_tag_reason=hidden_tag_reason,
    )


def batch_set_stock_tags(batch_set_stock_tags_model: BatchSetStockTagsModel):
    if not batch_set_stock_tags_model.entity_ids:
        return []

    tag_info = CreateTagInfoModel(
        tag=batch_set_stock_tags_model.tag,
        tag_reason=batch_set_stock_tags_model.tag_reason,
    )
    if not is_tag_info_existed(
        tag_info=tag_info, tag_type=batch_set_stock_tags_model.tag_type
    ):
        # 🧠 ML Signal: Conditional logic based on tag type.
        # 🧠 ML Signal: Model instantiation with multiple parameters.
        build_tag_info(tag_info=tag_info, tag_type=batch_set_stock_tags_model.tag_type)

    with contract_api.DBSession(provider="zvt", data_schema=StockTags)() as session:
        tag_type = batch_set_stock_tags_model.tag_type
        if tag_type == TagType.main_tag:
            main_tag = batch_set_stock_tags_model.tag
            stock_tags: List[StockTags] = StockTags.query_data(
                entity_ids=batch_set_stock_tags_model.entity_ids,
                filters=[StockTags.main_tag != main_tag],
                # 🧠 ML Signal: Function call with multiple parameters indicating complex logic.
                session=session,
                return_type="domain",
            )
        elif tag_type == TagType.sub_tag:
            sub_tag = batch_set_stock_tags_model.tag
            # ✅ Best Practice: Use of default mutable arguments (like lists) should be avoided; using None is a safer default.
            stock_tags: List[StockTags] = StockTags.query_data(
                # 🧠 ML Signal: Function call with specific parameters, indicating a pattern of usage.
                entity_ids=batch_set_stock_tags_model.entity_ids,
                filters=[StockTags.sub_tag != sub_tag],
                session=session,
                # ⚠️ SAST Risk (Low): Ensure that the session is properly managed to avoid stale data.
                return_type="domain",
            )
        # 🧠 ML Signal: Querying data from a database or data source.
        # 🧠 ML Signal: Conversion of data to a list, indicating a common data processing pattern.
        elif tag_type == TagType.hidden_tag:
            hidden_tag = batch_set_stock_tags_model.tag
            stock_tags: List[StockTags] = StockTags.query_data(
                entity_ids=batch_set_stock_tags_model.entity_ids,
                # 需要sqlite3版本>=3.37.0
                # ✅ Best Practice: Type hinting for better code readability and maintainability.
                filters=[
                    func.json_extract(StockTags.active_hidden_tags, f'$."{hidden_tag}"')
                    == None
                ],
                session=session,
                return_type="domain",
            )

        # ✅ Best Practice: Dictionary comprehension for efficient mapping.
        for stock_tag in stock_tags:
            tag_parameter: TagParameter = build_tag_parameter(
                tag_type=tag_type,
                # ✅ Best Practice: Type hinting for better code readability and maintainability.
                tag=batch_set_stock_tags_model.tag,
                tag_reason=batch_set_stock_tags_model.tag_reason,
                # 🧠 ML Signal: Logging information, useful for understanding code execution flow.
                stock_tag=stock_tag,
            )
            if tag_type == TagType.hidden_tag:
                active_hidden_tags = {
                    batch_set_stock_tags_model.tag: batch_set_stock_tags_model.tag_reason
                }
            # ✅ Best Practice: Type hinting for better code readability and maintainability.
            # 🧠 ML Signal: Logging information, useful for understanding code execution flow.
            # 🧠 ML Signal: Function call with specific parameters, indicating a pattern of usage.
            else:
                active_hidden_tags = stock_tag.active_hidden_tags

            set_stock_tags_model = SetStockTagsModel(
                entity_id=stock_tag.entity_id,
                main_tag=tag_parameter.main_tag,
                main_tag_reason=tag_parameter.main_tag_reason,
                sub_tag=tag_parameter.sub_tag,
                sub_tag_reason=tag_parameter.sub_tag_reason,
                active_hidden_tags=active_hidden_tags,
            )
            # 🧠 ML Signal: Function call with specific parameters, indicating a pattern of usage.

            build_stock_tags(
                # ✅ Best Practice: Use of a data model for structured data handling.
                set_stock_tags_model=set_stock_tags_model,
                timestamp=now_pd_timestamp(),
                # 🧠 ML Signal: Default behavior when no entity_ids are provided
                set_by_user=True,
                keep_current=False,
            )
            session.refresh(stock_tag)
        return stock_tags


# 🧠 ML Signal: Iterating over entity_ids to perform operations


# 🧠 ML Signal: Function call with specific parameters, indicating a pattern of usage.
def build_default_main_tag(entity_ids=None, force_rebuild=False):
    """
    build default main tag by industry

    :param entity_ids: entity ids
    :param force_rebuild: always rebuild it if True otherwise only build which not existed
    """
    if not entity_ids:
        entity_ids = get_entity_ids_by_filter(
            provider="em", ignore_delist=True, ignore_st=False, ignore_new_stock=False
        )

    df_block = Block.query_data(provider="em", filters=[Block.category == "industry"])
    industry_codes = df_block["code"].tolist()
    block_stocks: List[BlockStock] = BlockStock.query_data(
        provider="em",
        filters=[
            BlockStock.code.in_(industry_codes),
            BlockStock.stock_id.in_(entity_ids),
        ],
        return_type="domain",
    )
    entity_id_block_mapping = {
        block_stock.stock_id: block_stock for block_stock in block_stocks
    }

    for entity_id in entity_ids:
        stock_tags: List[StockTags] = StockTags.query_data(
            entity_id=entity_id, return_type="domain"
        )
        if not force_rebuild and stock_tags:
            logger.info(f"{entity_id} main tag has been set.")
            continue

        logger.info(f"build main tag for: {entity_id}")

        block_stock: BlockStock = entity_id_block_mapping.get(entity_id)
        if block_stock:
            main_tag = get_main_tag_by_industry(industry_name=block_stock.name)
            main_tag_reason = f"来自行业:{block_stock.name}"
        else:
            main_tag = "其他"
            main_tag_reason = "其他"

        build_stock_tags(
            set_stock_tags_model=SetStockTagsModel(
                entity_id=entity_id,
                main_tag=main_tag,
                main_tag_reason=main_tag_reason,
                sub_tag=None,
                sub_tag_reason=None,
                active_hidden_tags=None,
            ),
            timestamp=now_pd_timestamp(),
            # ✅ Best Practice: Consider using a dictionary to map tag_type to data_schema for better scalability and readability.
            set_by_user=False,
            keep_current=False,
        )


def build_default_sub_tags(entity_ids=None):
    if not entity_ids:
        entity_ids = get_entity_ids_by_filter(
            provider="em",
            ignore_delist=True,
            ignore_st=False,
            ignore_new_stock=False,
            # ⚠️ SAST Risk (Low): Using assert for control flow can be bypassed if Python is run with optimizations.
        )
    # 🧠 ML Signal: Function definition with specific parameters indicating a pattern for checking existence

    for entity_id in entity_ids:
        # 🧠 ML Signal: Dynamic schema retrieval based on tag type
        logger.info(f"build sub tag for: {entity_id}")
        # ⚠️ SAST Risk (Low): Potential for SQL injection if `data_schema` is not properly sanitized
        datas = StockTags.query_data(entity_id=entity_id, limit=1, return_type="domain")
        if not datas:
            raise AssertionError(f"Main tag must be set at first for {entity_id}")
        # 🧠 ML Signal: Querying data with specific filters

        current_stock_tags: StockTags = datas[0]
        keep_current = False
        if current_stock_tags.set_by_user:
            # ✅ Best Practice: Explicit check for truthiness
            logger.info(f"keep current tags set by user for: {entity_id}")
            keep_current = True

        # 🧠 ML Signal: Checking for existence before creation is a common pattern
        # ✅ Best Practice: Explicit return of False for clarity
        current_sub_tag = current_stock_tags.sub_tag
        filters = [BlockStock.stock_id == entity_id]
        # ⚠️ SAST Risk (Low): Potential information disclosure through error message
        if current_sub_tag:
            logger.info(f"{entity_id} current_sub_tag: {current_sub_tag}")
            # 🧠 ML Signal: Dynamic schema retrieval based on type
            current_sub_tags = current_stock_tags.sub_tags.keys()
            filters = filters + [BlockStock.name.notin_(current_sub_tags)]
        # ✅ Best Practice: Using context manager for database session ensures proper resource management
        # 🧠 ML Signal: Use of current timestamp for record creation

        df_block = Block.query_data(
            provider="em", filters=[Block.category == "concept"]
        )
        concept_codes = df_block["code"].tolist()
        filters = filters + [BlockStock.code.in_(concept_codes)]

        block_stocks: List[BlockStock] = BlockStock.query_data(
            provider="em",
            # 🧠 ML Signal: Hardcoded entity_id, could indicate a default or admin action
            # 🧠 ML Signal: Constructing a database model instance
            filters=filters,
            return_type="domain",
        )
        if not block_stocks:
            # 🧠 ML Signal: Function definition with parameters indicating a pattern for data processing
            logger.info(f"no block_stocks for: {entity_id}")
            continue
        # ⚠️ SAST Risk (Low): Hardcoded provider name "zvt" could lead to inflexibility or misconfiguration
        # ✅ Best Practice: Adding new record to session before committing
        # ✅ Best Practice: Committing session to persist changes
        # ⚠️ SAST Risk (Low): Hardcoded entity_id "admin" could lead to privilege escalation if misused

        for block_stock in block_stocks:
            sub_tag = block_stock.name
            if sub_tag in get_sub_tags():
                sub_tag_reason = f"来自概念:{sub_tag}"

                main_tag = get_main_tag_by_sub_tag(sub_tag)
                # ✅ Best Practice: Refreshing session to get updated state of the object
                # ✅ Best Practice: Use of a conversion function for timestamp ensures consistent data format
                # 🧠 ML Signal: Returning the created database object
                # ⚠️ SAST Risk (Low): Potential for ID collision if stock_pool_name is not unique
                main_tag_reason = sub_tag_reason
                if (main_tag == "其他" or not main_tag) and current_stock_tags.main_tag:
                    # 🧠 ML Signal: Use of model attributes to set object properties
                    main_tag = current_stock_tags.main_tag
                    main_tag_reason = current_stock_tags.main_tag_reason
                # 🧠 ML Signal: Function definition with parameters indicating a pattern for model building

                build_stock_tags(
                    # 🧠 ML Signal: Pattern of adding and committing to a session
                    # ⚠️ SAST Risk (Low): Use of a hardcoded provider name in DBSession
                    set_stock_tags_model=SetStockTagsModel(
                        # ✅ Best Practice: Refreshing the session to ensure the object is updated with the latest database state
                        # 🧠 ML Signal: Conditional check for existence in a list
                        # 🧠 ML Signal: Function call with model creation and timestamp
                        entity_id=entity_id,
                        main_tag=main_tag,
                        main_tag_reason=main_tag_reason,
                        sub_tag=sub_tag,
                        sub_tag_reason=sub_tag_reason,
                        active_hidden_tags=current_stock_tags.active_hidden_tags,
                        # 🧠 ML Signal: Returning a database object after creation
                    ),
                    # 🧠 ML Signal: String formatting for ID creation
                    timestamp=now_pd_timestamp(),
                    set_by_user=False,
                    keep_current=keep_current,
                )
            else:
                logger.info(f"ignore {sub_tag} not in sub_tag_info yet")


# 🧠 ML Signal: Querying data with filters
def get_tag_info_schema(tag_type: TagType):
    if tag_type == TagType.main_tag:
        data_schema = MainTagInfo
    elif tag_type == TagType.sub_tag:
        data_schema = SubTagInfo
    # 🧠 ML Signal: Conditional logic based on query results
    elif tag_type == TagType.hidden_tag:
        data_schema = HiddenTagInfo
    # 🧠 ML Signal: Conditional logic for different insert modes
    else:
        assert False

    return data_schema


def is_tag_info_existed(tag_info: CreateTagInfoModel, tag_type: TagType):
    # ✅ Best Practice: Use of set to avoid duplicate entries
    data_schema = get_tag_info_schema(tag_type=tag_type)
    with contract_api.DBSession(provider="zvt", data_schema=data_schema)() as session:
        # 🧠 ML Signal: Object creation with multiple attributes
        current_tags_info = data_schema.query_data(
            session=session,
            filters=[data_schema.tag == tag_info.tag],
            return_type="domain",
            # 🧠 ML Signal: Function definition with a specific task related to stock pool management
        )
        if current_tags_info:
            # ✅ Best Practice: Using a context manager for database session ensures proper resource management
            # 🧠 ML Signal: Querying data from a database using specific filters
            return True
        return False


def build_tag_info(tag_info: CreateTagInfoModel, tag_type: TagType):
    """
    Create tags info
    """
    # 🧠 ML Signal: Return statement indicating function output
    # ⚠️ SAST Risk (Medium): Potential risk of SQL injection if filters are not properly sanitized
    if is_tag_info_existed(tag_info=tag_info, tag_type=tag_type):
        raise HTTPException(
            status_code=409, detail=f"This tag has been registered in {tag_type}"
        )

    # ✅ Best Practice: Deleting an object from the session before committing
    data_schema = get_tag_info_schema(tag_type=tag_type)
    # ✅ Best Practice: Use of context manager for session ensures proper resource management
    with contract_api.DBSession(provider="zvt", data_schema=data_schema)() as session:
        timestamp = current_date()
        entity_id = "admin"
        tag_info_db = data_schema(
            id=f"admin_{tag_info.tag}",
            entity_id=entity_id,
            timestamp=timestamp,
            tag=tag_info.tag,
            tag_reason=tag_info.tag_reason,
        )
        session.add(tag_info_db)
        # 🧠 ML Signal: Use of type hints can be used to infer data structures and types
        session.commit()
        session.refresh(tag_info_db)
        return tag_info_db


def build_stock_pool_info(
    create_stock_pool_info_model: CreateStockPoolInfoModel, timestamp
):
    with contract_api.DBSession(provider="zvt", data_schema=StockPoolInfo)() as session:
        stock_pool_info = StockPoolInfo(
            entity_id="admin",
            timestamp=to_pd_timestamp(timestamp),
            id=f"admin_{create_stock_pool_info_model.stock_pool_name}",
            stock_pool_type=create_stock_pool_info_model.stock_pool_type.value,
            stock_pool_name=create_stock_pool_info_model.stock_pool_name,
            # 🧠 ML Signal: Use of flatten_list indicates data transformation patterns
        )
        # 🧠 ML Signal: Use of dictionary comprehensions for mapping
        session.add(stock_pool_info)
        session.commit()
        session.refresh(stock_pool_info)
        return stock_pool_info


def build_stock_pool(
    create_stock_pools_model: CreateStockPoolsModel, target_date=current_date()
):
    with contract_api.DBSession(provider="zvt", data_schema=StockPools)() as session:
        if create_stock_pools_model.stock_pool_name not in get_stock_pool_names():
            build_stock_pool_info(
                CreateStockPoolInfoModel(
                    # 🧠 ML Signal: Use of dictionary comprehensions for mapping
                    stock_pool_type=StockPoolType.custom,
                    stock_pool_name=create_stock_pools_model.stock_pool_name,
                ),
                timestamp=target_date,
            )
        # 🧠 ML Signal: Use of dictionary comprehensions for mapping
        # one instance per day
        stock_pool_id = f"admin_{to_time_str(target_date)}_{create_stock_pools_model.stock_pool_name}"
        datas: List[StockPools] = StockPools.query_data(
            session=session,
            filters=[
                StockPools.timestamp == to_pd_timestamp(target_date),
                StockPools.stock_pool_name == create_stock_pools_model.stock_pool_name,
            ],
            return_type="domain",
            # ⚠️ SAST Risk (Low): Potential NoneType access if entity_id is not found in entity_map
        )
        if datas:
            # ⚠️ SAST Risk (Low): Potential NoneType access if entity_id is not found in entity_map
            stock_pool = datas[0]
            if create_stock_pools_model.insert_mode == InsertMode.overwrite:
                stock_pool.entity_ids = create_stock_pools_model.entity_ids
            else:
                # ⚠️ SAST Risk (Low): Potential NoneType access if entity_id is not found in entity_tags_map
                stock_pool.entity_ids = list(
                    set(stock_pool.entity_ids + create_stock_pools_model.entity_ids)
                )
        else:
            stock_pool = StockPools(
                # 🧠 ML Signal: Checking for the presence of sub_tags can indicate data completeness or quality.
                entity_id="admin",
                timestamp=to_pd_timestamp(target_date),
                # ⚠️ SAST Risk (Low): Logging potentially sensitive information (entity_id).
                id=stock_pool_id,
                stock_pool_name=create_stock_pools_model.stock_pool_name,
                # 🧠 ML Signal: Use of fill_dict indicates data merging patterns
                entity_ids=create_stock_pools_model.entity_ids,
                # 🧠 ML Signal: Accessing a specific sub_tag from a collection.
            )
        session.add(stock_pool)
        # 🧠 ML Signal: Mapping sub_tag to its reason, indicating a relationship between data points.
        session.commit()
        session.refresh(stock_pool)
        # 🧠 ML Signal: Function call to derive main_tag from sub_tag, indicating a transformation or mapping.
        return stock_pool


# 🧠 ML Signal: Conditional logic to handle specific cases, such as "其他".


def del_stock_pool(stock_pool_name: str):
    with contract_api.DBSession(provider="zvt", data_schema=StockPoolInfo)() as session:
        stock_pool_info = StockPoolInfo.query_data(
            session=session,
            filters=[StockPoolInfo.stock_pool_name == stock_pool_name],
            return_type="domain",
            # ✅ Best Practice: Using a model to encapsulate data, improving readability and maintainability.
        )

        contract_api.del_data(
            data_schema=StockPools,
            filters=[StockPools.stock_pool_name == stock_pool_name],
        )

        if stock_pool_info:
            session.delete(stock_pool_info[0])
            session.commit()
            # ⚠️ SAST Risk (Low): Logging potentially sensitive information (set_stock_tags_model).
            # ✅ Best Practice: Consider adding a docstring to describe the function's purpose and behavior
            return "success"
        return "not found"


# ✅ Best Practice: Use a context manager for database session to ensure proper resource management
# 🧠 ML Signal: Function call to build stock tags, indicating a data processing step.
# 🧠 ML Signal: Querying data from a database can indicate data retrieval patterns


def query_stock_tag_stats(query_stock_tag_stats_model: QueryStockTagStatsModel):
    with contract_api.DBSession(provider="zvt", data_schema=TagStats)() as session:
        datas = TagStats.query_data(
            session=session,
            filters=[
                TagStats.stock_pool_name == query_stock_tag_stats_model.stock_pool_name
            ],
            # 🧠 ML Signal: Iterating over database query results is a common pattern
            # 🧠 ML Signal: Function calls within loops can indicate batch processing patterns
            # ✅ Best Practice: Consider adding type hints for function parameters and return type for better readability and maintainability.
            order=TagStats.timestamp.desc(),
            limit=1,
            return_type="domain",
        )
        if not datas:
            return []
        # 🧠 ML Signal: Usage of query_data method with specific filters and columns can indicate common data access patterns.

        target_date = datas[0].timestamp
        # 🧠 ML Signal: Converting a DataFrame column to a list is a common pattern for data processing.

        tag_stats_list: List[dict] = TagStats.query_data(
            # 🧠 ML Signal: Function definition with specific parameter types can be used to infer usage patterns.
            # ✅ Best Practice: Logging information messages helps in tracking the flow and state of the application.
            session=session,
            filters=[
                # 🧠 ML Signal: Querying data with specific filters can indicate common data access patterns.
                TagStats.stock_pool_name == query_stock_tag_stats_model.stock_pool_name,
                # ✅ Best Practice: Early return pattern improves code readability by reducing nested blocks.
                # 🧠 ML Signal: Calling a function with specific parameters can indicate a pattern of usage or behavior.
                # 🧠 ML Signal: Converting query results to a list can indicate common data processing patterns.
                TagStats.timestamp == target_date,
            ],
            return_type="dict",
            order=TagStats.position.asc(),
        )
        # 🧠 ML Signal: Querying data with specific filters can indicate common data access patterns.

        if query_stock_tag_stats_model.query_type == TagStatsQueryType.simple:
            return tag_stats_list

        # 🧠 ML Signal: List comprehension usage can indicate common data transformation patterns.
        entity_ids = flatten_list(
            [tag_stats["entity_ids"] for tag_stats in tag_stats_list]
        )
        # 🧠 ML Signal: Function definition with specific model parameter type

        # ⚠️ SAST Risk (Low): Logging information about empty results could potentially expose sensitive data.
        # get stocks meta
        stocks = Stock.query_data(
            provider="em", entity_ids=entity_ids, return_type="domain"
        )
        # ⚠️ SAST Risk (Low): Potential SQL injection if sub_tags are not properly sanitized
        entity_map = {item.entity_id: item for item in stocks}

        # 🧠 ML Signal: Function call with specific parameters can indicate common usage patterns.
        # get stock tags
        # ⚠️ SAST Risk (Low): Potential SQL injection if sub_tag is not properly sanitized
        tags_dict = StockTags.query_data(
            session=session,
            filters=[StockTags.entity_id.in_(entity_ids)],
            return_type="dict",
        )
        entity_tags_map = {item["entity_id"]: item for item in tags_dict}
        # ⚠️ SAST Risk (Low): Use of raw SQL functions can lead to SQL injection

        # get stock system tags
        system_tags_dict = StockSystemTags.query_data(
            session=session,
            filters=[
                StockSystemTags.timestamp == target_date,
                StockSystemTags.entity_id.in_(entity_ids),
            ],
            # ✅ Best Practice: Use of logging for information tracking
            return_type="dict",
        )
        entity_system_tags_map = {item["entity_id"]: item for item in system_tags_dict}

        # ⚠️ SAST Risk (Medium): Potential SQL injection risk if `hidden_tag` is not properly sanitized before being used in the query.
        for tag_stats in tag_stats_list:
            # ⚠️ SAST Risk (Low): Committing changes to the database without validation
            # ✅ Best Practice: Consider using parameterized queries to prevent SQL injection.
            stock_details = []
            # 🧠 ML Signal: Tracking changes to entity_id in result
            # ✅ Best Practice: Using a context manager for the session ensures that resources are properly managed and released.
            for entity_id in tag_stats["entity_ids"]:
                stock_details_model = {
                    "entity_id": entity_id,
                    "main_tag": tag_stats["main_tag"],
                    "code": entity_map.get(entity_id).code,
                    # 🧠 ML Signal: Usage of query_data method with filters indicates a pattern for querying databases.
                    "name": entity_map.get(entity_id).name,
                    # ⚠️ SAST Risk (Medium): Use of `func.json_extract` with dynamic input can lead to SQL injection if not properly handled.
                }

                stock_tags = entity_tags_map.get(entity_id)
                stock_details_model["sub_tag"] = stock_tags["sub_tag"]
                if stock_tags["active_hidden_tags"] is not None:
                    # 🧠 ML Signal: Logging patterns can be used to train models for detecting logging practices.
                    stock_details_model["hidden_tags"] = stock_tags[
                        "active_hidden_tags"
                    ].keys()
                else:
                    stock_details_model["hidden_tags"] = None

                # 🧠 ML Signal: Function definition with a specific task related to database operations
                # ✅ Best Practice: Converting to a dictionary before modification ensures that the original data structure is not directly altered.
                stock_system_tags = entity_system_tags_map.get(entity_id)
                stock_details_model = fill_dict(stock_system_tags, stock_details_model)
                # ✅ Best Practice: Committing the session after changes ensures that the database is updated with the latest data.
                # ⚠️ SAST Risk (Low): Potential risk if `contract_api.DBSession` is not properly handling exceptions
                # 🧠 ML Signal: Querying a database using specific filters

                stock_details.append(stock_details_model)
            tag_stats["stock_details"] = stock_details

        return tag_stats_list


# ✅ Best Practice: Refreshing the session ensures that the object is updated with the latest data from the database.


def refresh_main_tag_by_sub_tag(stock_tag: StockTags, set_by_user=False) -> StockTags:
    # ✅ Best Practice: Checking if the query result is empty before proceeding
    if not stock_tag.sub_tags:
        logger.warning(f"{stock_tag.entity_id} has no sub_tags yet")
        # 🧠 ML Signal: Logging information about the operation
        return stock_tag

    # 🧠 ML Signal: Function definition with parameters indicating a creation operation
    sub_tag = stock_tag.sub_tag
    # 🧠 ML Signal: Calling a function to perform an operation based on the query result
    sub_tag_reason = stock_tag.sub_tags[sub_tag]
    # 🧠 ML Signal: Instantiation of a model object with specific attributes

    # ⚠️ SAST Risk (Low): Deleting data from the database; ensure proper authorization and validation
    main_tag = get_main_tag_by_sub_tag(sub_tag)
    # 🧠 ML Signal: Conditional check for existence of an entity
    main_tag_reason = sub_tag_reason
    # ⚠️ SAST Risk (Low): Committing changes to the database; ensure atomicity and error handling
    # ✅ Best Practice: Using a context manager for database session ensures proper resource management.
    if main_tag == "其他":
        # 🧠 ML Signal: Function call to build or create an entity
        # 🧠 ML Signal: Querying data based on a specific tag can indicate a pattern of interest in certain industries.
        main_tag = stock_tag.main_tag
        main_tag_reason = stock_tag.main_tag_reason

    set_stock_tags_model = SetStockTagsModel(
        entity_id=stock_tag.entity_id,
        main_tag=main_tag,
        main_tag_reason=main_tag_reason,
        # 🧠 ML Signal: Function definition with a specific pattern of input and output
        sub_tag=sub_tag,
        # ✅ Best Practice: Returning a dictionary with clear keys improves readability and usability of the function's output.
        sub_tag_reason=sub_tag_reason,
        # 🧠 ML Signal: Querying a database using a session object
        # ⚠️ SAST Risk (Low): Potential risk if 'contract_api.DBSession' is not properly handling exceptions
        active_hidden_tags=stock_tag.active_hidden_tags,
    )
    logger.info(f"set_stock_tags_model:{set_stock_tags_model}")

    return build_stock_tags(
        set_stock_tags_model=set_stock_tags_model,
        timestamp=stock_tag.timestamp,
        set_by_user=set_by_user,
        # 🧠 ML Signal: Usage of a database session pattern
        keep_current=False,
        # ✅ Best Practice: Returning a dictionary with clear key-value pairs
    )


# 🧠 ML Signal: Function call to create a tag if it doesn't exist


# 🧠 ML Signal: Querying data with specific filters
def refresh_all_main_tag_by_sub_tag():
    with contract_api.DBSession(provider="zvt", data_schema=StockTags)() as session:
        stock_tags = StockTags.query_data(
            session=session,
            return_type="domain",
        )
        for stock_tag in stock_tags:
            refresh_main_tag_by_sub_tag(stock_tag)


# 🧠 ML Signal: Modifying attributes of queried data


def reset_to_default_main_tag(current_main_tag: str):
    df = StockTags.query_data(
        filters=[StockTags.main_tag == current_main_tag],
        # ⚠️ SAST Risk (Low): Committing changes to the database without error handling
        # 🧠 ML Signal: Querying data with specific filters
        columns=[StockTags.entity_id],
        return_type="df",
    )
    entity_ids = df["entity_id"].tolist()
    # ✅ Best Practice: Using a context manager for the session ensures it is properly closed after use.
    if not entity_ids:
        logger.info(f"all stocks with main_tag: {current_main_tag} has been reset")
        return
    # 🧠 ML Signal: Function call to create a main tag if it doesn't exist indicates a pattern of ensuring data integrity.
    # 🧠 ML Signal: Modifying attributes of queried data
    build_default_main_tag(entity_ids=entity_ids, force_rebuild=True)


# ⚠️ SAST Risk (Low): Committing changes to the database without error handling
# 🧠 ML Signal: Querying data with specific filters shows a pattern of data retrieval based on conditions.


def activate_industry_list(industry_list: List[str]):
    df_block = Block.query_data(
        provider="em",
        filters=[Block.category == "industry", Block.name.in_(industry_list)],
    )
    industry_codes = df_block["code"].tolist()
    block_stocks: List[BlockStock] = BlockStock.query_data(
        provider="em",
        filters=[BlockStock.code.in_(industry_codes)],
        # 🧠 ML Signal: Modifying data based on conditions is a common pattern in data processing.
        return_type="domain",
    )
    entity_ids = [block_stock.stock_id for block_stock in block_stocks]

    if not entity_ids:
        # ⚠️ SAST Risk (Low): Committing changes to the database without exception handling could lead to data inconsistency.
        # 🧠 ML Signal: Querying data with specific filters shows a pattern of data retrieval based on conditions.
        logger.info(f"No stocks in {industry_list}")
        return

    build_default_main_tag(entity_ids=entity_ids, force_rebuild=True)


# 🧠 ML Signal: Usage of a function to ensure a main tag exists before proceeding


def activate_sub_tags(activate_sub_tags_model: ActivateSubTagsModel):
    # ⚠️ SAST Risk (Low): Committing changes to the database without exception handling could lead to data inconsistency.
    # 🧠 ML Signal: Querying data with specific filters
    # 🧠 ML Signal: Modifying data based on conditions is a common pattern in data processing.
    # ⚠️ SAST Risk (Low): Potential SQL injection if filters are not properly sanitized
    sub_tags = activate_sub_tags_model.sub_tags
    with contract_api.DBSession(provider="zvt", data_schema=StockTags)() as session:
        result = {}
        for sub_tag in sub_tags:
            # df = StockTags.query_data(
            #     session=session,
            # 🧠 ML Signal: Building a tag parameter for each stock tag
            #     filters=[StockTags.sub_tag != sub_tag],
            #     columns=[StockTags.entity_id],
            #     return_type="df",
            # )
            # entity_ids = df["entity_id"].tolist()
            entity_ids = None
            # 🧠 ML Signal: Creating a model to set stock tags

            # stock_tag with sub_tag but not set to related main_tag yet
            stock_tags = StockTags.query_data(
                session=session,
                entity_ids=entity_ids,
                # 需要sqlite3版本>=3.37.0
                filters=[
                    func.json_extract(StockTags.sub_tags, f'$."{sub_tag}"') != None
                ],
                return_type="domain",
            )
            if not stock_tags:
                logger.info(f"all stocks with sub_tag: {sub_tag} has been activated")
                continue
            for stock_tag in stock_tags:
                stock_tag.sub_tag = sub_tag
                # 🧠 ML Signal: Building stock tags with specific parameters
                session.commit()
                session.refresh(stock_tag)
                result[stock_tag.entity_id] = refresh_main_tag_by_sub_tag(
                    stock_tag, set_by_user=True
                )
        return result


# ⚠️ SAST Risk (Low): Direct execution of code in the main block
# ✅ Best Practice: Using __all__ to define public API of the module
# ✅ Best Practice: Refreshing session to ensure data consistency
# ⚠️ SAST Risk (Low): Potentially unsafe print statement for debugging


def remove_hidden_tag(hidden_tag: str):
    with contract_api.DBSession(provider="zvt", data_schema=StockTags)() as session:
        stock_tags = StockTags.query_data(
            session=session,
            # 需要sqlite3版本>=3.37.0
            filters=[
                func.json_extract(StockTags.hidden_tags, f'$."{hidden_tag}"') != None
            ],
            return_type="domain",
        )
        if not stock_tags:
            logger.info(f"all stocks with hidden_tag: {hidden_tag} has been removed")
            return []
        for stock_tag in stock_tags:
            hidden_tags = dict(stock_tag.hidden_tags)
            hidden_tags.pop(hidden_tag)
            stock_tag.hidden_tags = hidden_tags
            session.commit()
            session.refresh(stock_tag)
        return stock_tags


def del_hidden_tag(tag: str):
    with contract_api.DBSession(provider="zvt", data_schema=HiddenTagInfo)() as session:
        hidden_tag_info = HiddenTagInfo.query_data(
            session=session,
            filters=[HiddenTagInfo.tag == tag],
            return_type="domain",
        )
        if not hidden_tag_info:
            logger.info(f"hidden_tag: {tag} has been removed")
            return []

        result = remove_hidden_tag(hidden_tag=tag)
        session.delete(hidden_tag_info[0])
        session.commit()
        return result


def _create_main_tag_if_not_existed(main_tag, main_tag_reason):
    main_tag_info = CreateTagInfoModel(tag=main_tag, tag_reason=main_tag_reason)
    if not is_tag_info_existed(tag_info=main_tag_info, tag_type=TagType.main_tag):
        build_tag_info(tag_info=main_tag_info, tag_type=TagType.main_tag)


def get_main_tag_industry_relation(main_tag):
    with contract_api.DBSession(provider="zvt", data_schema=StockTags)() as session:
        df = IndustryInfo.query_data(
            session=session,
            columns=[IndustryInfo.industry_name],
            filters=[IndustryInfo.main_tag == main_tag],
            return_type="df",
        )
        return {"main_tag": main_tag, "industry_list": df["industry_name"].tolist()}


def get_main_tag_sub_tag_relation(main_tag):
    with contract_api.DBSession(provider="zvt", data_schema=StockTags)() as session:
        df = SubTagInfo.query_data(
            session=session,
            columns=[SubTagInfo.tag],
            filters=[SubTagInfo.main_tag == main_tag],
            return_type="df",
        )
        return {"main_tag": main_tag, "sub_tag_list": df["tag"].tolist()}


def build_main_tag_industry_relation(
    main_tag_industry_relation: MainTagIndustryRelation,
):
    with contract_api.DBSession(provider="zvt", data_schema=StockTags)() as session:
        main_tag = main_tag_industry_relation.main_tag
        _create_main_tag_if_not_existed(main_tag=main_tag, main_tag_reason=main_tag)

        industry_list = main_tag_industry_relation.industry_list

        datas: List[IndustryInfo] = IndustryInfo.query_data(
            session=session,
            filters=[
                IndustryInfo.main_tag == main_tag,
                IndustryInfo.industry_name.notin_(industry_list),
            ],
            return_type="domain",
        )
        for data in datas:
            data.main_tag = "其他"
        session.commit()

        industry_info_list: List[IndustryInfo] = IndustryInfo.query_data(
            session=session,
            filters=[IndustryInfo.industry_name.in_(industry_list)],
            return_type="domain",
        )
        for industry_info in industry_info_list:
            industry_info.main_tag = main_tag
        session.commit()


def build_main_tag_sub_tag_relation(main_tag_sub_tag_relation: MainTagSubTagRelation):
    with contract_api.DBSession(provider="zvt", data_schema=SubTagInfo)() as session:
        main_tag = main_tag_sub_tag_relation.main_tag
        _create_main_tag_if_not_existed(main_tag=main_tag, main_tag_reason=main_tag)

        sub_tag_list = main_tag_sub_tag_relation.sub_tag_list

        datas: List[SubTagInfo] = SubTagInfo.query_data(
            session=session,
            filters=[
                SubTagInfo.main_tag == main_tag,
                SubTagInfo.tag.notin_(sub_tag_list),
            ],
            return_type="domain",
        )
        for data in datas:
            data.main_tag = "其他"
        session.commit()

        sub_tag_info_list: List[SubTagInfo] = SubTagInfo.query_data(
            session=session,
            filters=[SubTagInfo.tag.in_(sub_tag_list)],
            return_type="domain",
        )
        for sub_tag_info in sub_tag_info_list:
            sub_tag_info.main_tag = main_tag
        session.commit()


def change_main_tag(change_main_tag_model: ChangeMainTagModel):
    new_main_tag = change_main_tag_model.new_main_tag
    _create_main_tag_if_not_existed(main_tag=new_main_tag, main_tag_reason=new_main_tag)
    with contract_api.DBSession(provider="zvt", data_schema=StockTags)() as session:
        stock_tags: List[StockTags] = StockTags.query_data(
            filters=[StockTags.main_tag == change_main_tag_model.current_main_tag],
            session=session,
            return_type="domain",
        )

        for stock_tag in stock_tags:
            tag_parameter: TagParameter = build_tag_parameter(
                tag_type=TagType.main_tag,
                tag=new_main_tag,
                tag_reason=new_main_tag,
                stock_tag=stock_tag,
            )
            set_stock_tags_model = SetStockTagsModel(
                entity_id=stock_tag.entity_id,
                main_tag=tag_parameter.main_tag,
                main_tag_reason=tag_parameter.main_tag_reason,
                sub_tag=tag_parameter.sub_tag,
                sub_tag_reason=tag_parameter.sub_tag_reason,
                active_hidden_tags=stock_tag.active_hidden_tags,
            )

            build_stock_tags(
                set_stock_tags_model=set_stock_tags_model,
                timestamp=now_pd_timestamp(),
                set_by_user=True,
                keep_current=False,
            )
            session.refresh(stock_tag)
        return stock_tags


if __name__ == "__main__":
    print(del_hidden_tag(tag="妖"))
    # activate_industry_list(industry_list=["半导体"])
    # activate_sub_tags(ActivateSubTagsModel(sub_tags=["航天概念", "天基互联", "北斗导航", "通用航空"]))


# the __all__ is generated
__all__ = [
    "get_stock_tag_options",
    "build_stock_tags",
    "build_tag_parameter",
    "batch_set_stock_tags",
    "build_default_main_tag",
    "build_default_sub_tags",
    "get_tag_info_schema",
    "is_tag_info_existed",
    "build_tag_info",
    "build_stock_pool_info",
    "build_stock_pool",
    "query_stock_tag_stats",
    "refresh_main_tag_by_sub_tag",
    "refresh_all_main_tag_by_sub_tag",
    "reset_to_default_main_tag",
    "activate_industry_list",
    "activate_sub_tags",
]
