from zvt.api.kdata import get_kdata
from zvt.contract import IntervalLevel
from zvt.contract.api import get_db_session
from ..context import init_test_context

# 🧠 ML Signal: Initialization of test context, indicating a setup phase in the code
init_test_context()

# 🧠 ML Signal: Database session creation with specific provider and database name
day_k_session = get_db_session(provider="joinquant", db_name="stock_1d_kdata")  # type: sqlalchemy.orm.Session
# 🧠 ML Signal: Function name suggests a test, indicating a pattern of testing or validation.
# ⚠️ SAST Risk (Low): Potential exposure of database configuration details

day_1h_session = get_db_session(provider="joinquant", db_name="stock_1h_kdata")  # type: sqlalchemy.orm.Session

# 🧠 ML Signal: Usage of specific entity_id and provider indicates a pattern of data retrieval for a specific stock.
# 🧠 ML Signal: Database session creation with specific provider and database name
# ⚠️ SAST Risk (Low): Potential exposure of database configuration details
# ✅ Best Practice: Consider using constants or configuration for repeated literal values like entity_id and provider.

def test_jq_603220_kdata():
    df = get_kdata(
        entity_id="stock_sh_603220", session=day_k_session, level=IntervalLevel.LEVEL_1DAY, provider="joinquant"
    # ⚠️ SAST Risk (Low): Printing data frames can expose sensitive data in logs.
    # 🧠 ML Signal: Repeated pattern of data retrieval with different parameters indicates a pattern of multi-level data analysis.
    # ✅ Best Practice: Consider using constants or configuration for repeated literal values like entity_id and provider.
    )
    print(df)
    df = get_kdata(
        entity_id="stock_sh_603220", session=day_1h_session, level=IntervalLevel.LEVEL_1HOUR, provider="joinquant"
    )
    print(df)