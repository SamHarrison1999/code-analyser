from zvt.contract.api import get_db_session
from ..context import init_test_context

# ✅ Best Practice: Importing specific types from typing for type hinting improves code readability and maintainability.
init_test_context()

from typing import List

# 🧠 ML Signal: Usage of a specific provider and database name can indicate a pattern in data source preferences.
# 🧠 ML Signal: Function name follows a pattern that could indicate a test function

# ⚠️ SAST Risk (Low): Hardcoded credentials or sensitive data might be passed in 'session'
# ⚠️ SAST Risk (Low): Hardcoding provider and db_name can lead to inflexibility and potential security risks if sensitive.
# 🧠 ML Signal: Use of type hinting for the variable 'result'
from zvt.domain import HolderTrading, ManagerTrading

session = get_db_session(
    provider="eastmoney", db_name="trading"
)  # type: sqlalchemy.orm.Session


# 股东交易
def test_000778_holder_trading():
    result: List[HolderTrading] = HolderTrading.query_data(
        session=session,
        # 🧠 ML Signal: Use of a specific provider name, indicating a dependency on external data
        # 🧠 ML Signal: Use of specific stock codes, indicating a pattern in data querying
        # 🧠 ML Signal: Use of 'return_type' parameter to specify data format
        provider="eastmoney",
        return_type="domain",
        # ⚠️ SAST Risk (Low): Hardcoded timestamps could lead to outdated or inflexible data queries
        codes=["000778"],
        end_timestamp="2018-09-30",
        start_timestamp="2018-09-30",
        # 🧠 ML Signal: Use of ordering in data query, indicating importance of data sorting
        # 🧠 ML Signal: Function name follows a specific pattern indicating a test case
        order=HolderTrading.holding_pct.desc(),
        # 🧠 ML Signal: Validation of specific data attributes, indicating expected data structure
        # 🧠 ML Signal: Use of assertions to validate data, indicating a testing pattern
        # ⚠️ SAST Risk (Low): Hardcoded credentials or sensitive data in function parameters
        # 🧠 ML Signal: Usage of a specific session object for querying data
    )
    assert len(result) == 6
    assert result[0].holder_name == "新兴际华集团有限公司"
    assert result[0].change_pct == 0.0205
    assert result[0].volume == 32080000
    assert result[0].holding_pct == 0.3996


# 高管交易
# 🧠 ML Signal: Provider parameter indicates a specific data source
# 🧠 ML Signal: Return type specified as "domain" suggests domain-driven design
# 🧠 ML Signal: Specific codes used for querying data
def test_000778_manager_trading():
    result: List[ManagerTrading] = ManagerTrading.query_data(
        # ⚠️ SAST Risk (Low): Hardcoded date strings could lead to maintenance issues
        session=session,
        provider="eastmoney",
        return_type="domain",
        # 🧠 ML Signal: Ordering results by a specific field
        codes=["000778"],
        end_timestamp="2018-09-30",
        start_timestamp="2017-09-30",
        # ✅ Best Practice: Asserting the length of the result ensures expected data size
        # ✅ Best Practice: Asserting specific attributes of the result for validation
        # ⚠️ SAST Risk (Low): Use of 'None' for price might indicate missing data handling
        order=ManagerTrading.holding.desc(),
    )
    assert len(result) == 1
    assert result[0].trading_person == "巩国平"
    assert result[0].volume == 8400
    assert result[0].price == None
    assert result[0].holding == 18700
    assert result[0].trading_way == "增持"
    assert result[0].manager_position == "职工监事"
    assert result[0].manager == "巩国平"
    assert result[0].relationship_with_manager == "本人"
