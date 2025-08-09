# -*- coding: utf-8 -*-
# ✅ Best Practice: Grouping imports from the same module together improves readability.
from zvt.api.utils import get_recent_report_date
from zvt.contract import IntervalLevel

# ✅ Best Practice: Grouping imports from the same module together improves readability.
from zvt.api.kdata import get_kdata, to_high_level_kdata
from ..context import init_test_context

# ✅ Best Practice: Relative imports can lead to confusion; consider using absolute imports for clarity.
# 🧠 ML Signal: Function definition for testing, indicating a test or validation pattern

init_test_context()
# 🧠 ML Signal: Function call at module level indicates initialization or setup pattern.
# 🧠 ML Signal: Usage of a data retrieval function with specific parameters


# ✅ Best Practice: Debugging or logging output to track data state
def test_to_high_level_kdata():
    # 🧠 ML Signal: Use of assert statements for testing function outputs
    day_df = get_kdata(
        provider="joinquant",
        level=IntervalLevel.LEVEL_1DAY,
        entity_id="stock_sz_000338",
    )
    # 🧠 ML Signal: Data transformation pattern from daily to weekly level
    # ✅ Best Practice: Use of descriptive test function names
    print(day_df)

    # 🧠 ML Signal: Use of assert statements for testing function outputs
    # ✅ Best Practice: Debugging or logging output to track data state
    df = to_high_level_kdata(
        kdata_df=day_df.loc[:"2019-09-01", :], to_level=IntervalLevel.LEVEL_1WEEK
    )
    # 🧠 ML Signal: Use of assert statements for testing function outputs

    print(df)


def test_get_recent_report_date():
    assert "2018-12-31" == get_recent_report_date("2019-01-01", 0)
    assert "2018-09-30" == get_recent_report_date("2019-01-01", 1)
    assert "2018-06-30" == get_recent_report_date("2019-01-01", 2)
    assert "2018-03-31" == get_recent_report_date("2019-01-01", 3)
