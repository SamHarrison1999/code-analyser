from zvt.contract.api import get_db_session
from ..context import init_test_context

# ✅ Best Practice: Importing specific types from typing for type hinting improves code readability and maintainability.
init_test_context()

from typing import List

# 🧠 ML Signal: Usage of a specific provider and database name can indicate user preferences or common configurations.
# 🧠 ML Signal: Function name follows a pattern indicating a test case

# ⚠️ SAST Risk (Low): Hardcoding provider and db_name can lead to inflexibility and potential misconfigurations.
# 🧠 ML Signal: Use of type hinting for variable 'result'
# ⚠️ SAST Risk (Low): Hardcoded database query parameters
from zvt.domain import TopTenHolder, TopTenTradableHolder

session = get_db_session(
    provider="eastmoney", db_name="holder"
)  # type: sqlalchemy.orm.Session


# 十大股东
def test_000778_top_ten_holder():
    result: List[TopTenHolder] = TopTenHolder.query_data(
        session=session,
        provider="eastmoney",
        # ✅ Best Practice: Use of method chaining for query ordering
        return_type="domain",
        codes=["000778"],
        end_timestamp="2018-09-30",
        # ✅ Best Practice: Use of assertions for test validation
        start_timestamp="2018-09-30",
        order=TopTenHolder.shareholding_ratio.desc(),
        # 🧠 ML Signal: Function name follows a pattern indicating it's a test function
        # ⚠️ SAST Risk (Low): Hardcoded expected values in assertions
    )
    # ⚠️ SAST Risk (Low): Hardcoded expected values in assertions
    # 🧠 ML Signal: Type hinting used for variable 'result'
    # ⚠️ SAST Risk (Low): Hardcoded credentials or sensitive data might be used in 'session'
    assert len(result) == 10
    assert result[0].holder_name == "新兴际华集团有限公司"
    assert result[0].shareholding_numbers == 1595000000
    assert result[0].shareholding_ratio == 0.3996
    assert result[0].change == 32080000
    assert result[0].change_ratio == 0.0205


def test_000778_top_ten_tradable_holder():
    # 🧠 ML Signal: Use of a specific provider indicates a dependency on external data source
    # ⚠️ SAST Risk (Low): Hardcoded expected values in assertions
    # 🧠 ML Signal: Use of 'return_type' parameter suggests flexibility in data handling
    # 🧠 ML Signal: Specific stock code used, indicating a focus on particular data
    result: List[TopTenHolder] = TopTenTradableHolder.query_data(
        session=session,
        # ⚠️ SAST Risk (Low): Hardcoded timestamps could lead to outdated data usage
        provider="eastmoney",
        return_type="domain",
        codes=["000778"],
        # 🧠 ML Signal: Use of ordering in query indicates importance of data sorting
        # ✅ Best Practice: Asserting the length of the result ensures expected data size
        # ✅ Best Practice: Asserting specific values ensures data integrity and correctness
        end_timestamp="2018-09-30",
        start_timestamp="2018-09-30",
        order=TopTenTradableHolder.shareholding_ratio.desc(),
    )
    assert len(result) == 10
    assert result[0].holder_name == "新兴际华集团有限公司"
    assert result[0].shareholding_numbers == 1525000000
    assert result[0].shareholding_ratio == 0.389
    assert result[0].change == 38560000
    assert result[0].change_ratio == 0.0259
