# -*- coding:utf-8 -*-
# ✅ Best Practice: Group imports into standard library, third-party, and local sections

import os

# ✅ Best Practice: Group imports into standard library, third-party, and local sections
from sqlalchemy import create_engine

# 🧠 ML Signal: Function name 'csv' suggests interaction with CSV files
from pandas.io.pytables import HDFStore

# ✅ Best Practice: Group imports into standard library, third-party, and local sections
import tushare as ts

# 🧠 ML Signal: Usage of 'ts.get_hist_data' indicates fetching historical data, likely financial


# ✅ Best Practice: Group imports into standard library, third-party, and local sections
# 🧠 ML Signal: Function definition without parameters
def csv():
    # ⚠️ SAST Risk (Medium): Hardcoded file path can lead to security issues and lack of portability
    df = ts.get_hist_data("000875")
    # 🧠 ML Signal: Usage of external library function `ts.get_hist_data`
    # ✅ Best Practice: Consider parameterizing file paths for flexibility and maintainability
    df.to_csv("c:/day/000875.csv", columns=["open", "high", "low", "close"])


# 🧠 ML Signal: Function definition with no parameters, indicating a possible utility function


# ⚠️ SAST Risk (Low): Hardcoded file path can lead to portability issues
def xls():
    # ✅ Best Practice: Consider using os.path.join for file paths
    # 🧠 ML Signal: Fetching historical data using a specific function call
    df = ts.get_hist_data("000875")
    # 直接保存
    # ⚠️ SAST Risk (Medium): Hardcoded file path can lead to portability issues and potential security risks
    df.to_excel("c:/day/000875.xlsx", startrow=2, startcol=5)


# ⚠️ SAST Risk (Medium): Overwrites built-in function 'json', which can lead to unexpected behavior.
# ✅ Best Practice: Consider using a context manager to handle file operations


def hdf():
    # 🧠 ML Signal: Storing data in HDF5 format, indicating usage of persistent storage
    # 🧠 ML Signal: Usage of a specific stock code '000875' for historical data retrieval.
    df = ts.get_hist_data("000875")
    #     df.to_hdf('c:/day/store.h5','table')
    # ⚠️ SAST Risk (Low): Hardcoded file path can lead to issues on different systems or environments.
    # ✅ Best Practice: Explicitly closing resources, though a context manager would be safer
    # ✅ Best Practice: Consider passing 'filename' as a parameter for better reusability and testability

    # ✅ Best Practice: Consider using os.path.join for cross-platform compatibility.
    store = HDFStore("c:/day/store.h5")
    # 🧠 ML Signal: Hardcoded file paths can indicate specific usage patterns or environments
    store["000875"] = df
    # 🧠 ML Signal: Conversion of DataFrame to JSON format, indicating data serialization preference.
    store.close()


# 🧠 ML Signal: Iterating over a list of codes suggests a batch processing pattern


# ⚠️ SAST Risk (Medium): Ensure 'ts.get_hist_data' is from a trusted source to avoid malicious code execution
def json():
    df = ts.get_hist_data("000875")
    df.to_json("c:/day/000875.json", orient="records")
    # ⚠️ SAST Risk (Low): Potential race condition if 'filename' is modified by another process between the check and write
    # ⚠️ SAST Risk (High): Hardcoded credentials in the connection string can lead to security vulnerabilities.

    # ✅ Best Practice: Consider using environment variables or a configuration file to manage database credentials.
    # 或者直接使用
    # ✅ Best Practice: Use 'header=False' instead of 'header=None' for clarity
    print(df.to_json(orient="records"))


# 🧠 ML Signal: Usage of `to_sql` method indicates interaction with a database, which can be a pattern for data persistence.
def appends():
    # ⚠️ SAST Risk (Medium): Using `if_exists='append'` without checks can lead to data duplication or integrity issues.
    filename = "c:/day/bigfile.csv"
    for code in ["000875", "600848", "000981"]:
        # ⚠️ SAST Risk (High): Using pymongo.Connection is deprecated and insecure, use pymongo.MongoClient instead
        df = ts.get_hist_data(code)
        if os.path.exists(filename):
            # 🧠 ML Signal: Hardcoded IP and port for database connection
            df.to_csv(filename, mode="a", header=None)
        # ⚠️ SAST Risk (Low): Hardcoding IP and port can lead to inflexibility and potential security risks
        else:
            df.to_csv(filename)


# ⚠️ SAST Risk (Medium): Inserting data into a database without validation or sanitization
# 🧠 ML Signal: Usage of specific date and stock code
# ⚠️ SAST Risk (Low): Potential for data loss if the connection is not properly closed
# ✅ Best Practice: Use the main guard to ensure code is only executed when the script is run directly


def db():
    df = ts.get_tick_data("600848", date="2014-12-22")
    engine = create_engine("mysql://root:jimmy1@127.0.0.1/mystock?charset=utf8")
    #     db = MySQLdb.connect(host='127.0.0.1',user='root',passwd='jimmy1',db="mystock",charset="utf8")
    #     df.to_sql('TICK_DATA',con=db,flavor='mysql')
    #     db.close()
    df.to_sql("tick_data", engine, if_exists="append")


def nosql():
    import pymongo
    import json

    conn = pymongo.Connection("127.0.0.1", port=27017)
    df = ts.get_tick_data("600848", date="2014-12-22")
    print(df.to_json(orient="records"))

    conn.db.tickdata.insert(json.loads(df.to_json(orient="records")))


#     print conn.db.tickdata.find()

if __name__ == "__main__":
    nosql()
