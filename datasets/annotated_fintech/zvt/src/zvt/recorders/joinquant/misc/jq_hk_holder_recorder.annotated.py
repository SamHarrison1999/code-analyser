import pandas as pd

# 🧠 ML Signal: Importing specific functions from a module indicates selective usage patterns
from jqdatapy.api import run_query

# 🧠 ML Signal: Importing specific functions from a module indicates selective usage patterns
from zvt.contract.api import df_to_db, get_data
from zvt.contract.recorder import TimestampsDataRecorder

# 🧠 ML Signal: Importing specific classes from a module indicates selective usage patterns
from zvt.domain import Index
from zvt.domain.misc.holder import HkHolder

# 🧠 ML Signal: Importing specific classes from a module indicates selective usage patterns
from zvt.recorders.joinquant.common import to_entity_id
from zvt.utils.pd_utils import pd_is_not_null

# 🧠 ML Signal: Importing specific classes from a module indicates selective usage patterns
from zvt.utils.time_utils import to_time_str, TIME_FORMAT_DAY, to_pd_timestamp

# ✅ Best Practice: Class attributes should be documented to explain their purpose and usage.

# 🧠 ML Signal: Importing specific functions from a module indicates selective usage patterns

# ✅ Best Practice: Class attributes should be documented to explain their purpose and usage.
# 这里选择继承TimestampsDataRecorder是因为
# 🧠 ML Signal: Importing specific functions from a module indicates selective usage patterns
# 1)时间上就是交易日的列表,这个是可知的，可以以此为增量计算点
# 🧠 ML Signal: Importing specific functions from a module indicates selective usage patterns
# ✅ Best Practice: Class attributes should be documented to explain their purpose and usage.
# 2)HkHolder数据结构的设计：
# 沪股通/深股通 每日 持有 标的(股票)的情况
# 抓取的角度是entity从Index中获取 沪股通/深股通，然后按 每日 去获取


class JoinquantHkHolderRecorder(TimestampsDataRecorder):
    entity_provider = "exchange"
    entity_schema = Index

    # ✅ Best Practice: Use of default values for function parameters improves usability and reduces errors.
    provider = "joinquant"
    # 🧠 ML Signal: Use of hardcoded values can indicate specific business logic or domain-specific constants.
    data_schema = HkHolder

    def __init__(
        self,
        day_data=False,
        force_update=False,
        sleeping_time=5,
        real_time=False,
        start_timestamp=None,
        end_timestamp=None,
    ) -> None:
        # 聚宽编码
        # ✅ Best Practice: Consider adding a docstring to describe the purpose and parameters of the function
        # 市场通编码	市场通名称
        # 310001	沪股通
        # ✅ Best Practice: Calling the superclass's __init__ method ensures proper initialization of the base class.
        # 🧠 ML Signal: Usage of pandas date_range to generate a list of business days
        # ⚠️ SAST Risk (High): Use of eval() can lead to code injection vulnerabilities if input is not properly sanitized.
        # 310002	深股通
        # ✅ Best Practice: Use a consistent date format for better readability and maintainability
        # 310003	港股通（沪）
        # 🧠 ML Signal: Dynamic method invocation using eval() indicates a pattern of flexible code execution.
        # 🧠 ML Signal: Use of external data fetching function with filters and parameters.
        # 🧠 ML Signal: Filtering data based on entity attributes.
        # 310004	港股通（深）
        codes = ["310001", "310002"]

        super().__init__(
            force_update,
            sleeping_time,
            ["cn"],
            None,
            codes,
            # 🧠 ML Signal: Use of provider pattern for data access.
            # 🧠 ML Signal: Use of data schema for structured data retrieval.
            day_data,
            # 🧠 ML Signal: Ordering data dynamically.
            real_time=real_time,
            fix_duplicate_way="ignore",
            # 🧠 ML Signal: Limiting data retrieval to a specific number of records.
            start_timestamp=start_timestamp,
            # 🧠 ML Signal: Iterating over timestamps to process data for each timestamp
            end_timestamp=end_timestamp,
            # 🧠 ML Signal: Specifying return type for data retrieval.
            # ⚠️ SAST Risk (Low): Potential SQL injection risk if `entity.code` or `timestamp` are not sanitized
        )

    def init_timestamps(self, entity):
        # 🧠 ML Signal: Use of session for database operations.
        # 聚宽数据从2017年3月17开始
        return pd.date_range(
            start=to_pd_timestamp("2017-3-17"), end=pd.Timestamp.now(), freq="B"
        ).tolist()

    # ✅ Best Practice: Check if records list is not empty before accessing its elements.
    # ✅ Best Practice: Consider using logging instead of print for better control over output

    # 覆盖这个方式是因为，HkHolder里面entity其实是股票，而recorder中entity是 Index类型(沪股通/深股通)
    def get_latest_saved_record(self, entity):
        # 🧠 ML Signal: Checking if DataFrame is not null before processing
        order = eval(
            "self.data_schema.{}.desc()".format(self.get_evaluated_time_field())
        )
        # ✅ Best Practice: Use rename with a dictionary for clarity and maintainability

        records = get_data(
            # ✅ Best Practice: Ensure consistent datetime format for timestamps
            filters=[HkHolder.holder_code == entity.code],
            provider=self.provider,
            data_schema=self.data_schema,
            order=order,
            limit=1,
            return_type="domain",
            # 🧠 ML Signal: Mapping entity codes to entity IDs
            # 🧠 ML Signal: Processing and modifying stock codes
            session=self.session,
        )
        # 🧠 ML Signal: Creating unique IDs for records
        if records:
            # 🧠 ML Signal: Running a recorder with a specified sleeping time
            # ⚠️ SAST Risk (Low): Ensure `df_to_db` handles data safely to prevent SQL injection
            # ✅ Best Practice: Use `if __name__ == "__main__":` to ensure code only runs when script is executed directly
            # ✅ Best Practice: Use `__all__` to define public API of the module
            return records[0]
        return None

    def record(self, entity, start, end, size, timestamps):
        for timestamp in timestamps:
            df = run_query(
                table="finance.STK_HK_HOLD_INFO",
                conditions=f"link_id#=#{entity.code}&day#=#{to_time_str(timestamp)}",
            )
            print(df)

            if pd_is_not_null(df):
                df.rename(
                    columns={
                        "day": "timestamp",
                        "link_id": "holder_code",
                        "link_name": "holder_name",
                    },
                    inplace=True,
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"])

                df["entity_id"] = df["code"].apply(
                    lambda x: to_entity_id(entity_type="stock", jq_code=x)
                )
                df["code"] = df["code"].apply(lambda x: x.split(".")[0])

                # id格式为:{holder_name}_{entity_id}_{timestamp}
                df["id"] = df[["holder_name", "entity_id", "timestamp"]].apply(
                    lambda se: "{}_{}_{}".format(
                        se["holder_name"],
                        se["entity_id"],
                        to_time_str(se["timestamp"], fmt=TIME_FORMAT_DAY),
                    ),
                    axis=1,
                )

                df_to_db(
                    df=df,
                    data_schema=self.data_schema,
                    provider=self.provider,
                    force_update=self.force_update,
                )


if __name__ == "__main__":
    JoinquantHkHolderRecorder(sleeping_time=10).run()


# the __all__ is generated
__all__ = ["JoinquantHkHolderRecorder"]
