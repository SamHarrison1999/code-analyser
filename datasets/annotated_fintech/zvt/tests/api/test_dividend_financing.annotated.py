from ..context import init_test_context

# ✅ Best Practice: Ensure the test context is initialized before importing other modules to avoid dependency issues.
init_test_context()

from zvt.domain import SpoDetail, RightsIssueDetail, DividendFinancing
from zvt.contract.api import get_db_session

# 🧠 ML Signal: Function name pattern indicates a test function
# 🧠 ML Signal: Usage of a specific provider and database name can indicate user preferences or common configurations.
from zvt.utils.time_utils import to_pd_timestamp

# 🧠 ML Signal: Querying data with specific parameters
# 🧠 ML Signal: Use of session object for database operations
# ⚠️ SAST Risk (Low): Potential exposure of sensitive data through query parameters

session = get_db_session(
    provider="eastmoney", db_name="dividend_financing"
)  # type: sqlalchemy.orm.Session


# 增发详情
def test_000778_spo_detial():
    result = SpoDetail.query_data(
        session=session,
        # 🧠 ML Signal: Hardcoded provider name
        # 🧠 ML Signal: Specifying return type for query
        provider="eastmoney",
        return_type="domain",
        # 🧠 ML Signal: Querying with specific codes
        codes=["000778"],
        end_timestamp="2018-09-30",
        # 🧠 ML Signal: Use of specific timestamp for data filtering
        order=SpoDetail.timestamp.desc(),
    )
    # 🧠 ML Signal: Ordering query results by timestamp
    assert len(result) == 4
    # 🧠 ML Signal: Assertion to check the length of the result
    # 🧠 ML Signal: Type hinting for variable
    # 🧠 ML Signal: Use of a specific query pattern with parameters
    # 🧠 ML Signal: Use of session and provider parameters in data query
    latest: SpoDetail = result[0]
    assert latest.timestamp == to_pd_timestamp("2017-04-01")
    assert latest.spo_issues == 347600000
    assert latest.spo_price == 5.15
    assert latest.spo_raising_fund == 1766000000


# 配股详情
# 🧠 ML Signal: Assertion to check specific attributes of the result
# 🧠 ML Signal: Use of ordering in query
def test_000778_rights_issue_detail():
    result = RightsIssueDetail.query_data(
        session=session,
        # ⚠️ SAST Risk (Low): Potential for assertion to fail if data changes
        provider="eastmoney",
        return_type="domain",
        # ✅ Best Practice: Type hinting for better code readability and maintainability
        codes=["000778"],
        # 🧠 ML Signal: Function name follows a pattern that could indicate a test case
        end_timestamp="2018-09-30",
        # 🧠 ML Signal: Use of a query method to retrieve data
        # ⚠️ SAST Risk (Low): Potential for SQL injection if inputs are not sanitized
        # 🧠 ML Signal: Use of session object indicates a database interaction
        # ⚠️ SAST Risk (Low): Potential for assertion to fail if data changes
        order=RightsIssueDetail.timestamp.desc(),
    )
    assert len(result) == 2
    latest: RightsIssueDetail = result[0]
    assert latest.timestamp == to_pd_timestamp("2001-09-10")
    assert latest.rights_issues == 43570000
    assert latest.rights_raising_fund == 492300000
    assert latest.rights_issue_price == 11.3


# ⚠️ SAST Risk (Low): Potential for assertion to fail if data changes


# 🧠 ML Signal: Use of order by clause in query
# 分红融资
def test_000778_dividend_financing():
    result = DividendFinancing.query_data(
        # 🧠 ML Signal: Use of assert statements for validation
        session=session,
        # ✅ Best Practice: Type hinting for better code readability and maintainability
        provider="eastmoney",
        return_type="domain",
        codes=["000778"],
        end_timestamp="2018-09-30",
        order=DividendFinancing.timestamp.desc(),
    )
    assert len(result) == 22
    latest: DividendFinancing = result[1]
    assert latest.timestamp == to_pd_timestamp("2017")
    assert latest.dividend_money == 598632026.4
    assert latest.spo_issues == 347572815.0
    assert latest.rights_issues == 0
    assert latest.ipo_issues == 0
