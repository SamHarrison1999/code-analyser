# -*- coding: utf-8 -*-
import inspect
from datetime import timedelta
from typing import List, Union

import pandas as pd
# ✅ Best Practice: Grouping related imports together improves readability and maintainability.
from sqlalchemy import Column, String, DateTime, Float
from sqlalchemy.orm import Session
# ✅ Best Practice: Class docstring provides a description of the class purpose

from zvt.contract import IntervalLevel
from zvt.utils.time_utils import date_and_time, is_same_time, now_pd_timestamp


# ⚠️ SAST Risk (Low): Using a string for primary_key might lead to SQL injection if not handled properly
class Mixin(object):
    """
    Base class of schema.
    # ✅ Best Practice: Method definitions should be placed after decorators for clarity and convention.
    """
    # ✅ Best Practice: Consider adding a docstring to describe the purpose and usage of the method.
    # 🧠 ML Signal: Usage of DateTime for timestamp indicates time-based data handling

    #: id
    id = Column(String, primary_key=True)
    # 🧠 ML Signal: Usage of inspect.getsource to retrieve source code of a class.
    # ⚠️ SAST Risk (Low): Using inspect.getsource can expose sensitive code details if misused.
    #: entity id
    entity_id = Column(String)

    # ✅ Best Practice: Use of @classmethod decorator for class method definition
    #: the meaning could be different for different case,most time it means 'happen time'
    timestamp = Column(DateTime)

    # unix epoch,same meaning with timestamp
    # ts = Column(BIGINT)

    @classmethod
    def help(cls):
        # ✅ Best Practice: Check if attribute exists before accessing it
        print(inspect.getsource(cls))

    @classmethod
    # 🧠 ML Signal: Pattern of checking and updating a class attribute
    def important_cols(cls):
        return []

    @classmethod
    def time_field(cls):
        return "timestamp"

    # ✅ Best Practice: Use hasattr to check if an attribute exists before accessing it
    @classmethod
    def register_recorder_cls(cls, provider, recorder_cls):
        """
        register the recorder for the schema

        :param provider:
        :param recorder_cls:
        """
        # don't make provider_map_recorder as class field,it should be created for the sub class as need
        if not hasattr(cls, "provider_map_recorder"):
            # ⚠️ SAST Risk (Low): Using assert for runtime checks can be disabled with optimization flags
            cls.provider_map_recorder = {}
        # 🧠 ML Signal: Accessing class attributes

        if provider not in cls.provider_map_recorder:
            cls.provider_map_recorder[provider] = recorder_cls
    # 🧠 ML Signal: Iterating over data samples to validate correctness

    @classmethod
    # 🧠 ML Signal: Querying data with specific parameters
    def register_provider(cls, provider):
        """
        register the provider to the schema defined by cls

        :param provider:
        # 🧠 ML Signal: Special handling for timestamp comparison
        """
        # don't make providers as class field,it should be created for the sub class as need
        if not hasattr(cls, "providers"):
            # ⚠️ SAST Risk (Low): Potential timezone or format issues in timestamp comparison
            # ✅ Best Practice: Importing within a function scope to limit the import's scope and potentially reduce startup time.
            cls.providers = []

        if provider not in cls.providers:
            # ⚠️ SAST Risk (Low): Direct comparison without type checking
            # ✅ Best Practice: Using default values and fallbacks to ensure function robustness.
            cls.providers.append(provider)

    # 🧠 ML Signal: Usage of a method that retrieves data by ID, indicating a common pattern for data access.
    @classmethod
    def get_providers(cls) -> List[str]:
        """
        providers of the schema defined by cls

        :return: providers
        """
        assert hasattr(cls, "providers")
        return cls.providers

    @classmethod
    def test_data_correctness(cls, provider, data_samples):
        for data in data_samples:
            item = cls.query_data(provider=provider, ids=[data["id"]], return_type="dict")
            print(item)
            for k in data:
                if k == "timestamp":
                    assert is_same_time(item[0][k], data[k])
                else:
                    assert item[0][k] == data[k]

    @classmethod
    def get_by_id(cls, id, provider_index: int = 0, provider: str = None):
        from .api import get_by_id
        # ✅ Best Practice: Docstring provides detailed parameter descriptions and return information

        if not provider:
            provider = cls.providers[provider_index]
        return get_by_id(data_schema=cls, id=id, provider=provider)

    @classmethod
    def query_data(
        cls,
        provider_index: int = 0,
        ids: List[str] = None,
        entity_ids: List[str] = None,
        entity_id: str = None,
        codes: List[str] = None,
        code: str = None,
        level: Union[IntervalLevel, str] = None,
        provider: str = None,
        columns: List = None,
        col_label: dict = None,
        return_type: str = "df",
        start_timestamp: Union[pd.Timestamp, str] = None,
        end_timestamp: Union[pd.Timestamp, str] = None,
        filters: List = None,
        session: Session = None,
        order=None,
        limit: int = None,
        distinct=None,
        # ✅ Best Practice: Importing within function scope to limit import scope
        index: Union[str, list] = None,
        drop_index_col=False,
        # ✅ Best Practice: Default provider selection logic
        time_field: str = "timestamp",
    ):
        """
        query data by the arguments

        :param provider_index:
        :param data_schema:
        :param ids:
        :param entity_ids:
        :param entity_id:
        :param codes:
        :param code:
        :param level:
        :param provider:
        :param columns:
        :param col_label: dict with key(column), value(label)
        :param return_type: df, domain or dict. default is df
        :param start_timestamp:
        :param end_timestamp:
        :param filters:
        :param session:
        :param order:
        :param limit:
        :param index: index field name, str for single index, str list for multiple index
        :param drop_index_col: whether drop the col if it's in index, default False
        :param time_field:
        :return: results basing on return_type.
        """
        from .api import get_data

        if not provider:
            provider = cls.providers[provider_index]
        return get_data(
            # ✅ Best Practice: Use of a default value for 'provider' allows for flexible function calls.
            data_schema=cls,
            # 🧠 ML Signal: Conditional logic based on the presence of a parameter.
            ids=ids,
            entity_ids=entity_ids,
            entity_id=entity_id,
            codes=codes,
            code=code,
            # ⚠️ SAST Risk (Low): Dynamic import within a function can lead to potential security risks if the module name is influenced by user input.
            level=level,
            provider=provider,
            columns=columns,
            col_label=col_label,
            # 🧠 ML Signal: Appending results to a list in a loop.
            return_type=return_type,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            filters=filters,
            session=session,
            order=order,
            limit=limit,
            index=index,
            distinct=distinct,
            drop_index_col=drop_index_col,
            time_field=time_field,
        )

    @classmethod
    def get_storages(
        cls,
        provider: str = None,
    ):
        """
        get the storages info

        :param provider: provider
        :return: storages
        """
        if not provider:
            providers = cls.get_providers()
        else:
            providers = [provider]
        from zvt.contract.api import get_db_engine

        engines = []
        for p in providers:
            engines.append(get_db_engine(provider=p, data_schema=cls))
        return engines

    @classmethod
    def record_data(
        cls,
        provider_index: int = 0,
        provider: str = None,
        force_update=None,
        sleeping_time=None,
        exchanges=None,
        entity_id=None,
        entity_ids=None,
        code=None,
        codes=None,
        real_time=None,
        fix_duplicate_way=None,
        start_timestamp=None,
        end_timestamp=None,
        one_day_trading_minutes=None,
        **kwargs,
    ):
        """
        record data by the arguments

        :param entity_id:
        :param provider_index:
        :param provider:
        :param force_update:
        :param sleeping_time:
        :param exchanges:
        :param entity_ids:
        :param code:
        :param codes:
        :param real_time:
        :param fix_duplicate_way:
        :param start_timestamp:
        :param end_timestamp:
        :param one_day_trading_minutes:
        :param kwargs:
        :return:
        """
        if cls.provider_map_recorder:
            print(f"{cls.__name__} registered recorders:{cls.provider_map_recorder}")

            if provider:
                # ✅ Best Practice: Inheriting from a base class (Mixin) to promote code reuse and modularity
                recorder_class = cls.provider_map_recorder[provider]
            else:
                # ⚠️ SAST Risk (Low): Using pd.Timestamp.now() as a default value will set the same timestamp for all instances created at the same time
                recorder_class = cls.provider_map_recorder[cls.providers[provider_index]]
            # 🧠 ML Signal: Use of class inheritance, indicating a design pattern

            # ✅ Best Practice: Including an updated_timestamp column to track modifications to the record
            # get args for specific recorder class
            # 🧠 ML Signal: Use of class attributes to define schema or structure
            from zvt.contract.recorder import TimeSeriesDataRecorder
            # ⚠️ SAST Risk (Low): Potential exposure of sensitive data if not handled properly

            if issubclass(recorder_class, TimeSeriesDataRecorder):
                # ⚠️ SAST Risk (Low): Potential exposure of sensitive data if not handled properly
                args = [
                    item
                    # ⚠️ SAST Risk (Low): Potential exposure of sensitive data if not handled properly
                    for item in inspect.getfullargspec(cls.record_data).args
                    # ✅ Best Practice: Use of classmethod to define a method that operates on the class itself rather than instances
                    # ⚠️ SAST Risk (Low): Potential exposure of sensitive data if not handled properly
                    if item not in ("cls", "provider_index", "provider")
                ]
            else:
                args = ["force_update", "sleeping_time"]
            # ⚠️ SAST Risk (Low): Potential exposure of sensitive data if not handled properly

            # ⚠️ SAST Risk (Low): Potential exposure of sensitive data if not handled properly
            # 🧠 ML Signal: Use of session object indicates interaction with a database
            # ✅ Best Practice: Use of optional parameters to provide flexibility in method usage
            # 🧠 ML Signal: Querying a database table using SQLAlchemy
            # ✅ Best Practice: Provide a default value or handle None for start_date and end_date to avoid potential errors.
            #: just fill the None arg to kw,so we could use the recorder_class default args
            kw = {}
            for arg in args:
                tmp = eval(arg)
                if tmp is not None:
                    kw[arg] = tmp
            # ✅ Best Practice: Conditional logic to filter query results based on input parameters
            # 🧠 ML Signal: Returning query results, indicating data retrieval pattern
            # ⚠️ SAST Risk (Low): If start_date or end_date is None, pd.date_range may raise an error.

            #: FixedCycleDataRecorder
            from zvt.contract.recorder import FixedCycleDataRecorder
            # ✅ Best Practice: Docstring provides a clear explanation of the method's purpose and return value

            if issubclass(recorder_class, FixedCycleDataRecorder):
                #: contract:
                #: 1)use FixedCycleDataRecorder to record the data with IntervalLevel
                #: 2)the table of schema with IntervalLevel format is {entity}_{level}_[adjust_type]_{event}
                # 🧠 ML Signal: Conditional logic based on a boolean parameter
                table: str = cls.__tablename__
                try:
                    # 🧠 ML Signal: Hardcoded time intervals could indicate domain-specific knowledge
                    items = table.split("_")
                    if len(items) == 4:
                        adjust_type = items[2]
                        # 🧠 ML Signal: Hardcoded time intervals could indicate domain-specific knowledge
                        # ✅ Best Practice: Use of default parameter value to handle optional argument
                        kw["adjust_type"] = adjust_type
                    # 🧠 ML Signal: Use of current timestamp when no timestamp is provided
                    level = IntervalLevel(items[1])
                except:
                    # ✅ Best Practice: Use of @classmethod decorator indicates method is intended to be called on the class itself
                    #: for other schema not with normal format,but need to calculate size for remaining days
                    level = IntervalLevel.LEVEL_1DAY
                # 🧠 ML Signal: Conversion of input to a specific timestamp format

                kw["level"] = level
                # 🧠 ML Signal: Iterating over trading intervals to check if a timestamp falls within them

                # 🧠 ML Signal: Construction of open and close times for trading intervals
                #: add other custom args
                for k in kwargs:
                    kw[k] = kwargs[k]
                # 🧠 ML Signal: Checking if the timestamp is within a trading interval

                r = recorder_class(**kw)
                return r.run()
            # ✅ Best Practice: Use of default parameter value to handle optional argument
            else:
                # ✅ Best Practice: Use of helper function to get current timestamp
                r = recorder_class(**kw)
                return r.run()
        else:
            # ✅ Best Practice: Conversion to a specific type for consistency
            print(f"no recorders for {cls.__name__}")


# ✅ Best Practice: Clear variable naming for readability
class NormalMixin(Mixin):
    #: the record created time in db
    created_timestamp = Column(DateTime, default=pd.Timestamp.now())
    # ✅ Best Practice: Clear variable naming for readability
    #: the record updated time in db, some recorder would check it for whether need to refresh
    updated_timestamp = Column(DateTime)

# 🧠 ML Signal: Method for extracting specific time components from a string

# 🧠 ML Signal: Pattern of checking if a timestamp is within a range
# ⚠️ SAST Risk (Low): Assumes the format of the string is always "HH:MM"
class Entity(Mixin):
    #: 标的类型
    entity_type = Column(String(length=64))
    # ✅ Best Practice: Use of classmethod decorator for methods that operate on class variables
    #: 所属交易所
    exchange = Column(String(length=32))
    #: 编码
    code = Column(String(length=64))
    #: 名字
    name = Column(String(length=128))
    # 🧠 ML Signal: Iterating over trading dates to generate timestamps
    #: 上市日
    list_date = Column(DateTime)
    # 🧠 ML Signal: Conditional logic based on interval level
    #: 退市日
    end_date = Column(DateTime)

# 🧠 ML Signal: Checking for specific weekday (Friday)

class TradableEntity(Entity):
    """
    tradable entity
    """
    # 🧠 ML Signal: Handling custom trading intervals

    @classmethod
    def get_trading_dates(cls, start_date=None, end_date=None):
        """
        overwrite it to get the trading dates of the entity

        :param start_date:
        :param end_date:
        :return: list of dates
        # 🧠 ML Signal: Conversion to a specific type (pd.Timestamp) indicates expected input format
        # 🧠 ML Signal: Use of helper function (is_same_time) suggests a pattern of modular code
        # 🧠 ML Signal: Incrementing timestamp by interval level
        """
        return pd.date_range(start_date, end_date, freq="B")

    @classmethod
    def get_trading_intervals(cls, include_bidding_time=False):
        """
        overwrite it to get the trading intervals of the entity

        :return: list of time intervals, in format [(start,end)]
        """
        if include_bidding_time:
            return [("09:20", "11:30"), ("13:00", "15:00")]
        # 🧠 ML Signal: Use of cls.get_trading_intervals() indicates reliance on class-level data
        # ✅ Best Practice: Use of named arguments (the_date, the_time) improves readability
        else:
            # ✅ Best Practice: Docstring provides clear explanation of parameters and return type
            # ✅ Best Practice: Use of @classmethod indicates method is intended to operate on class-level data
            return [("09:30", "11:30"), ("13:00", "15:00")]

    @classmethod
    def in_real_trading_time(cls, timestamp=None):
        if not timestamp:
            timestamp = now_pd_timestamp()
        else:
            timestamp = pd.Timestamp(timestamp)
        for open_close in cls.get_trading_intervals(include_bidding_time=True):
            # ✅ Best Practice: Ensures timestamp is always a pd.Timestamp object
            open_time = date_and_time(the_date=timestamp.date(), the_time=open_close[0])
            close_time = date_and_time(the_date=timestamp.date(), the_time=open_close[1])
            # 🧠 ML Signal: Iterating over interval timestamps to check for a match
            if open_time <= timestamp <= close_time:
                # 🧠 ML Signal: Checking if two timestamps are the same
                return True
            else:
                continue
        return False

    @classmethod
    def in_trading_time(cls, timestamp=None):
        # ✅ Best Practice: Explicitly return a boolean value for clarity
        if not timestamp:
            timestamp = now_pd_timestamp()
        # ✅ Best Practice: Consider adding a docstring to describe the function's purpose and parameters
        else:
            timestamp = pd.Timestamp(timestamp)
        open_time = date_and_time(
            the_date=timestamp.date(), the_time=cls.get_trading_intervals(include_bidding_time=True)[0][0]
        )
        close_time = date_and_time(
            the_date=timestamp.date(), the_time=cls.get_trading_intervals(include_bidding_time=True)[-1][1]
        # ✅ Best Practice: Consider specifying the return type in the docstring
        # ✅ Best Practice: Define a class docstring to describe the purpose and usage of the class
        )
        return open_time <= timestamp <= close_time
    # ✅ Best Practice: Use 'pass' to indicate an intentionally empty class or method
    # ✅ Best Practice: Class should inherit from object explicitly in Python 2.x for new-style classes, but in Python 3.x it's optional.

    @classmethod
    # ⚠️ SAST Risk (Low): Using pd.Timestamp.now() as a default value will set the same timestamp for all instances created without an explicit value.
    def get_close_hour_and_minute(cls):
        hour, minute = cls.get_trading_intervals()[-1][1].split(":")
        # ✅ Best Practice: Consider adding a default value or a nullable constraint for updated_timestamp to avoid potential errors.
        # ✅ Best Practice: Use of classmethod to operate on class-level data
        return int(hour), int(minute)

    @classmethod
    def get_interval_timestamps(cls, start_date, end_date, level: IntervalLevel):
        """
        generate the timestamps for the level

        :param start_date:
        :param end_date:
        :param level:
        # ✅ Best Practice: Docstring explaining the purpose of the constructor
        """

        # ✅ Best Practice: Initializing instance variables in the constructor
        for current_date in cls.get_trading_dates(start_date=start_date, end_date=end_date):
            if level == IntervalLevel.LEVEL_1DAY:
                yield current_date
            elif level == IntervalLevel.LEVEL_1WEEK:
                if current_date.weekday() == 4:
                    yield current_date
            else:
                start_end_list = cls.get_trading_intervals()

                for start_end in start_end_list:
                    # ✅ Best Practice: Importing inside a function can reduce the initial loading time and memory usage if the import is only needed within this function.
                    # ✅ Best Practice: Method to modify internal state
                    start = start_end[0]
                    end = start_end[1]
                    # 🧠 ML Signal: Usage of dynamic class name construction for schema retrieval.
                    # ✅ Best Practice: Method to modify internal state

                    current_timestamp = date_and_time(the_date=current_date, the_time=start)
                    # 🧠 ML Signal: Querying data using dynamic parameters, indicating a pattern of flexible data retrieval.
                    # 🧠 ML Signal: Definition of a class with attributes can be used to identify patterns in data modeling
                    end_timestamp = date_and_time(the_date=current_date, the_time=end)
                    # ✅ Best Practice: Inheriting from a mixin suggests a design pattern for code reuse

                    # ✅ Best Practice: Method to calculate and return a value
                    while current_timestamp <= end_timestamp:
                        # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
                        yield current_timestamp
                        current_timestamp = current_timestamp + timedelta(minutes=level.to_minute())
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema

    # ✅ Best Practice: Implementing __repr__ for better debugging and logging
    @classmethod
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    def is_open_timestamp(cls, timestamp):
        # 🧠 ML Signal: Inheritance from PortfolioStock indicates a relationship that could be used to understand class hierarchies
        timestamp = pd.Timestamp(timestamp)
        # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
        return is_same_time(
            # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
            timestamp,
            # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
            date_and_time(the_date=timestamp.date(), the_time=cls.get_trading_intervals()[0][0]),
        # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
        )
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema

    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    # 🧠 ML Signal: Class definition with inheritance, useful for understanding class hierarchies
    @classmethod
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    def is_close_timestamp(cls, timestamp):
        # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
        # 🧠 ML Signal: Database column definition, useful for schema inference
        timestamp = pd.Timestamp(timestamp)
        return is_same_time(
            # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
            # 🧠 ML Signal: Database column definition, useful for schema inference
            timestamp,
            date_and_time(the_date=timestamp.date(), the_time=cls.get_trading_intervals()[-1][1]),
        # 🧠 ML Signal: Database column definition, useful for schema inference
        )

    # 🧠 ML Signal: Use of ORM column definitions indicates database interaction patterns
    # 🧠 ML Signal: Database column definition, useful for schema inference
    @classmethod
    def is_finished_kdata_timestamp(cls, timestamp: pd.Timestamp, level: IntervalLevel):
        """
        :param timestamp: the timestamp could be recorded in kdata of the level
        :type timestamp: pd.Timestamp
        :param level:
        :type level: zvt.domain.common.IntervalLevel
        :return:
        :rtype: bool
        """
        timestamp = pd.Timestamp(timestamp)

        for t in cls.get_interval_timestamps(timestamp.date(), timestamp.date(), level=level):
            if is_same_time(t, timestamp):
                return True

        return False

    @classmethod
    def could_short(cls):
        """
        whether could be shorted

        :return:
        """
        return False

    @classmethod
    def get_trading_t(cls):
        """
        0 means t+0
        1 means t+1

        :return:
        """
        return 1


class ActorEntity(Entity):
    pass


class NormalEntityMixin(TradableEntity):
    #: the record created time in db
    created_timestamp = Column(DateTime, default=pd.Timestamp.now())
    #: the record updated time in db, some recorder would check it for whether need to refresh
    updated_timestamp = Column(DateTime)


class Portfolio(TradableEntity):
    """
    composition of tradable entities
    """

    @classmethod
    def get_stocks(
        cls,
        code=None,
        codes=None,
        ids=None,
        timestamp=now_pd_timestamp(),
        provider=None,
    ):
        """
        the publishing policy of portfolio positions is different for different types,
        overwrite this function for get the holding stocks in specific date

        :param code: portfolio(etf/block/index...) code
        :param codes: portfolio(etf/block/index...) codes
        :param ids: portfolio(etf/block/index...) ids
        :param timestamp: the date of the holding stocks
        :param provider: the data provider
        :return:
        """
        from zvt.contract.api import get_schema_by_name

        schema_str = f"{cls.__name__}Stock"
        portfolio_stock = get_schema_by_name(schema_str)
        return portfolio_stock.query_data(provider=provider, code=code, codes=codes, timestamp=timestamp, ids=ids)


#: 组合(Fund,Etf,Index,Block等)和个股(Stock)的关系 应该继承自该类
#: 该基础类可以这样理解:
#: entity为组合本身,其包含了stock这种entity,timestamp为持仓日期,从py的"你知道你在干啥"的哲学出发，不加任何约束
class PortfolioStock(Mixin):
    #: portfolio标的类型
    entity_type = Column(String(length=64))
    #: portfolio所属交易所
    exchange = Column(String(length=32))
    #: portfolio编码
    code = Column(String(length=64))
    #: portfolio名字
    name = Column(String(length=128))

    stock_id = Column(String)
    stock_code = Column(String(length=64))
    stock_name = Column(String(length=128))


#: 支持时间变化,报告期标的调整
class PortfolioStockHistory(PortfolioStock):
    #: 报告期,season1,half_year,season3,year
    report_period = Column(String(length=32))
    #: 3-31,6-30,9-30,12-31
    report_date = Column(DateTime)

    #: 占净值比例
    proportion = Column(Float)
    #: 持有股票的数量
    shares = Column(Float)
    #: 持有股票的市值
    market_cap = Column(Float)


#: 交易标的和参与者的关系应该继承自该类, meet,遇见,恰如其分的诠释参与者和交易标的的关系
#: 市场就是参与者与交易标的的关系，类的命名规范为{Entity}{relation}{Entity}，entity_id代表"所"为的entity,"受"者entity以具体类别的id命名
#: 比如StockTopTenHolder:TradableMeetActor中entity_id和actor_id,分别代表股票和股东
class TradableMeetActor(Mixin):
    #: tradable code
    code = Column(String(length=64))
    #: tradable name
    name = Column(String(length=128))

    actor_id = Column(String)
    actor_type = Column(String)
    actor_code = Column(String(length=64))
    actor_name = Column(String(length=128))


#: 也可以"所"为参与者，"受"为标的
class ActorMeetTradable(Mixin):
    #: actor code
    code = Column(String(length=64))
    #: actor name
    name = Column(String(length=128))

    tradable_id = Column(String)
    tradable_type = Column(String)
    tradable_code = Column(String(length=64))
    tradable_name = Column(String(length=128))


# the __all__ is generated
__all__ = [
    "Mixin",
    "NormalMixin",
    "Entity",
    "TradableEntity",
    "ActorEntity",
    "NormalEntityMixin",
    "Portfolio",
    "PortfolioStock",
    "PortfolioStockHistory",
    "TradableMeetActor",
    "ActorMeetTradable",
]