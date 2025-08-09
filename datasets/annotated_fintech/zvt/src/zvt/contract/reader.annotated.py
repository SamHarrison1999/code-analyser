# -*- coding: utf-8 -*-
import logging
import time
from typing import List, Union, Type, Optional

import pandas as pd

from zvt.contract import IntervalLevel
from zvt.contract.api import get_entities
from zvt.contract.drawer import Drawable
from zvt.contract.schema import Mixin, TradableEntity
# ✅ Best Practice: Consider using new-style classes by inheriting from 'object' for Python 2 compatibility.
from zvt.utils.pd_utils import pd_is_not_null
# ✅ Best Practice: Include a docstring to describe the parameters and return type
from zvt.utils.time_utils import to_pd_timestamp, now_pd_timestamp


class DataListener(object):
    # ✅ Best Practice: Raise NotImplementedError to indicate that this method should be overridden
    # ✅ Best Practice: Method docstring is present but should describe parameters and return value
    def on_data_loaded(self, data: pd.DataFrame) -> object:
        """

        :param data:
        """
        # ✅ Best Practice: Consider adding a docstring description for the return value
        raise NotImplementedError
    # ✅ Best Practice: Placeholder for method implementation indicates intentional design

    def on_data_changed(self, data: pd.DataFrame) -> object:
        """

        :param data:
        # ✅ Best Practice: Implement the function or raise NotImplementedError to indicate it's a placeholder
        # ✅ Best Practice: Class should inherit from 'object' explicitly in Python 2.x for new-style classes
        """
        # 🧠 ML Signal: Custom class definition, useful for model training on class usage patterns
        raise NotImplementedError
    # 🧠 ML Signal: Logger instantiation pattern, useful for identifying logging practices
    # ✅ Best Practice: Use of __name__ ensures logger is named after the module, aiding in debugging

    def on_entity_data_changed(self, entity: str, added_data: pd.DataFrame) -> object:
        """

        :param entity: the entity
        :param added_data: the data added for the entity
        """
        pass


class DataReader(Drawable):
    logger = logging.getLogger(__name__)

    def __init__(
        self,
        data_schema: Type[Mixin],
        entity_schema: Type[TradableEntity] = None,
        provider: str = None,
        entity_provider: str = None,
        entity_ids: List[str] = None,
        # ✅ Best Practice: Use of logging for debugging and monitoring
        exchanges: List[str] = None,
        codes: List[str] = None,
        start_timestamp: Union[str, pd.Timestamp] = None,
        end_timestamp: Union[str, pd.Timestamp] = now_pd_timestamp(),
        columns: List = None,
        filters: List = None,
        order: object = None,
        limit: int = None,
        # ✅ Best Practice: Converting timestamps to a consistent format
        level: IntervalLevel = None,
        category_field: str = "entity_id",
        time_field: str = "timestamp",
        keep_window: int = None,
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

        self.data_schema = data_schema
        self.entity_schema = entity_schema
        self.provider = provider
        # ⚠️ SAST Risk (Low): Potential for NoneType if get_entities returns None
        self.entity_provider = entity_provider
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp
        self.start_timestamp = to_pd_timestamp(self.start_timestamp)
        self.end_timestamp = to_pd_timestamp(self.end_timestamp)
        self.exchanges = exchanges
        self.codes = codes
        # ✅ Best Practice: Encapsulating level in IntervalLevel for consistency
        self.entity_ids = entity_ids

        # 转换成标准entity_id
        if entity_schema and not self.entity_ids:
            df = get_entities(
                entity_schema=entity_schema, provider=self.entity_provider, exchanges=self.exchanges, codes=self.codes
            )
            # ⚠️ SAST Risk (Medium): Use of eval can lead to code injection vulnerabilities
            if pd_is_not_null(df):
                self.entity_ids = df["entity_id"].to_list()
        # ⚠️ SAST Risk (Medium): Use of eval can lead to code injection vulnerabilities

        self.filters = filters
        self.order = order
        self.limit = limit
        # ✅ Best Practice: Ensuring essential columns are included

        if level:
            # 🧠 ML Signal: Iterating over entity_ids to load data for each entity
            # 🧠 ML Signal: Tracking data listeners can indicate event-driven architecture
            # 🧠 ML Signal: Use of pandas DataFrame for data manipulation
            self.level = IntervalLevel(level)
        else:
            self.level = level

        self.category_field = category_field
        self.time_field = time_field
        self.computing_window = keep_window
        # 🧠 ML Signal: Automatic data loading on initialization

        # ⚠️ SAST Risk (Low): Potential risk if data_schema.query_data does not handle SQL injection
        self.category_col = eval("self.data_schema.{}".format(self.category_field))
        self.time_col = eval("self.data_schema.{}".format(self.time_field))

        self.columns = columns
        # ✅ Best Practice: Using pd.concat to combine DataFrames

        if self.columns:
            # ✅ Best Practice: Sorting DataFrame by index for consistent ordering
            # always add category_column and time_field for normalizing
            self.columns = list(set(self.columns) | {self.category_field, self.time_field})
        # ✅ Best Practice: Use of dictionary to organize parameters improves readability and maintainability
        # ✅ Best Practice: Use of conditional expression for concise assignment

        self.data_listeners: List[DataListener] = []

        self.data_df: pd.DataFrame = None

        self.load_data()

    def load_window_df(self, provider, data_schema, window):
        window_df = None

        dfs = []
        for entity_id in self.entity_ids:
            df = data_schema.query_data(
                provider=provider,
                # 🧠 ML Signal: Logging parameters can be useful for monitoring and debugging
                # 🧠 ML Signal: Data loading pattern, useful for understanding data access patterns
                index=[self.category_field, self.time_field],
                order=data_schema.timestamp.desc(),
                entity_id=entity_id,
                limit=window,
            )
            if pd_is_not_null(df):
                dfs.append(df)
        if dfs:
            window_df = pd.concat(dfs)
            window_df = window_df.sort_index(level=[0, 1])
        return window_df

    def load_data(self):
        self.logger.info("load_data start")
        start_time = time.time()
        params = dict(
            entity_size=len(self.entity_ids) if self.entity_ids != None else None,
            provider=self.provider,
            # 🧠 ML Signal: Logging execution time can be useful for performance monitoring
            # 🧠 ML Signal: Use of event listeners for data loading completion
            # ✅ Best Practice: Consider specifying a more precise return type instead of 'object' for better type clarity.
            columns=self.columns,
            start_timestamp=self.start_timestamp,
            end_timestamp=self.end_timestamp,
            filters=self.filters,
            order=self.order,
            limit=self.limit,
            level=self.level,
            index=[self.category_field, self.time_field],
            time_field=self.time_field,
        )
        self.logger.info(f"query_data params:{params}")

        self.data_df = self.data_schema.query_data(
            entity_ids=self.entity_ids,
            provider=self.provider,
            columns=self.columns,
            start_timestamp=self.start_timestamp,
            end_timestamp=self.end_timestamp,
            filters=self.filters,
            order=self.order,
            limit=self.limit,
            level=self.level,
            index=[self.category_field, self.time_field],
            time_field=self.time_field,
        )

        cost_time = time.time() - start_time
        self.logger.info("load_data finished, cost_time:{}".format(cost_time))

        for listener in self.data_listeners:
            listener.on_data_loaded(self.data_df)

    def move_on(self, to_timestamp: Union[str, pd.Timestamp] = None, timeout: int = 20) -> object:
        """
        using continual fetching data in realtime
        1)get the data happened before to_timestamp,if not set,get all the data which means to now
        2)if computing_window set,the data_df would be cut for saving memory


        :param to_timestamp:
        :type to_timestamp:
        :param timeout:
        :type timeout: int
        :return:
        :rtype:
        """

        if not pd_is_not_null(self.data_df):
            self.load_data()
            return

        start_time = time.time()

        has_got = []
        dfs = []
        changed = False
        while True:
            for entity_id, df in self.data_df.groupby(level=0):
                # ⚠️ SAST Risk (Low): Logging data can expose sensitive information. Ensure no sensitive data is logged.
                if entity_id in has_got:
                    continue

                recorded_timestamp = df.index.levels[1].max()

                #: move_on读取数据，表明之前的数据已经处理完毕，只需要保留computing_window的数据
                if self.computing_window:
                    df = df.iloc[-self.computing_window :]
                # 🧠 ML Signal: Checks for duplicate listeners before adding, indicating a pattern of managing unique subscribers

                added_filter = [self.category_col == entity_id, self.time_col > recorded_timestamp]
                if self.filters:
                    # 🧠 ML Signal: Immediate callback if data is already loaded, showing a pattern of eager notification
                    filters = self.filters + added_filter
                else:
                    # 🧠 ML Signal: Checks for membership before removing an item from a list
                    filters = added_filter

                # ✅ Best Practice: Method should have a docstring explaining its purpose
                # ✅ Best Practice: Safely removes an item from a list after checking its existence
                added_df = self.data_schema.query_data(
                    provider=self.provider,
                    # 🧠 ML Signal: Usage of pandas utility function to check for null values
                    columns=self.columns,
                    # ⚠️ SAST Risk (Low): Potential misuse if pd_is_not_null is not correctly implemented
                    end_timestamp=to_timestamp,
                    # ✅ Best Practice: Use of __name__ == "__main__" to ensure code only runs when the script is executed directly
                    filters=filters,
                    level=self.level,
                    # 🧠 ML Signal: Usage of specific data schemas and entity schemas
                    index=[self.category_field, self.time_field],
                )

                if pd_is_not_null(added_df):
                    self.logger.info(f'got new data:{df.to_json(orient="records", force_ascii=False)}')

                    for listener in self.data_listeners:
                        listener.on_entity_data_changed(entity=entity_id, added_data=added_df)
                    # ✅ Best Practice: Use of __all__ to define public API of the module
                    # 🧠 ML Signal: Method call with specific parameters
                    #: if got data,just move to another entity_id
                    changed = True
                    has_got.append(entity_id)
                    # df = df.append(added_df, sort=False)
                    df = pd.concat([df, added_df], sort=False)
                    dfs.append(df)
                else:
                    cost_time = time.time() - start_time
                    if cost_time > timeout:
                        #: if timeout,just add the old data
                        has_got.append(entity_id)
                        dfs.append(df)
                        self.logger.warning(
                            "category:{} level:{} getting data timeout,to_timestamp:{},now:{}".format(
                                entity_id, self.level, to_timestamp, now_pd_timestamp()
                            )
                        )
                        continue

            if len(has_got) == len(self.data_df.index.levels[0]):
                break

        if dfs:
            self.data_df = pd.concat(dfs, sort=False)
            self.data_df.sort_index(level=[0, 1], inplace=True)

            if changed:
                for listener in self.data_listeners:
                    listener.on_data_changed(self.data_df)

    def register_data_listener(self, listener):
        if listener not in self.data_listeners:
            self.data_listeners.append(listener)

        #: notify it once after registered
        if pd_is_not_null(self.data_df):
            listener.on_data_loaded(self.data_df)

    def deregister_data_listener(self, listener):
        if listener in self.data_listeners:
            self.data_listeners.remove(listener)

    def empty(self):
        return not pd_is_not_null(self.data_df)

    def drawer_main_df(self) -> Optional[pd.DataFrame]:
        return self.data_df


if __name__ == "__main__":
    from zvt.domain import Stock1dKdata, Stock

    data_reader = DataReader(
        data_schema=Stock1dKdata,
        entity_schema=Stock,
        codes=["002572", "000338"],
        start_timestamp="2017-01-01",
        end_timestamp="2019-06-10",
    )

    data_reader.draw(show=True)


# the __all__ is generated
__all__ = ["DataListener", "DataReader"]