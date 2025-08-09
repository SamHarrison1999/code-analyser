# -*- coding: utf-8 -*-
# ✅ Best Practice: Import only necessary functions or classes to reduce memory usage and improve code clarity
import pandas as pd

from zvt.utils.pd_utils import drop_continue_duplicate

# 🧠 ML Signal: Usage of pd.Series to create a pandas Series from a list


# 🧠 ML Signal: Function call pattern with named argument
def test_drop_continue_duplicate():
    data1 = [1, 2, 2, 3, 4, 4, 5]
    # 🧠 ML Signal: Usage of assert to verify function output
    s = pd.Series(data=data1)
    s1 = drop_continue_duplicate(s=s)
    assert s1.tolist() == [1, 2, 3, 4, 5]
    # 🧠 ML Signal: Usage of pd.DataFrame to create a DataFrame from a dictionary

    data2 = [1, 2, 2, 2, 4, 4, 5]
    # ✅ Best Practice: Debugging aid with print statement
    # 🧠 ML Signal: Function call pattern with multiple named arguments
    # 🧠 ML Signal: Usage of assert to verify function output

    df = pd.DataFrame(data={"A": data1, "B": data2})
    print(df)
    df1 = drop_continue_duplicate(s=df, col="A")
    assert df1["A"].tolist() == [1, 2, 3, 4, 5]

    df2 = drop_continue_duplicate(s=df, col="B")
    assert df2["A"].tolist() == [1, 2, 4, 5]
