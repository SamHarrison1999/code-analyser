# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
NOTE:
- This scripts is a demo to import example data import Qlib
- !!!!!!!!!!!!!!!TODO!!!!!!!!!!!!!!!!!!!:
    - Its structure is not well designed and very ugly, your contribution is welcome to make importing dataset easier
"""
from datetime import date, datetime as dt
import os
from pathlib import Path
import random
import shutil
import time
import traceback

from arctic import Arctic, chunkstore
import arctic
from arctic import Arctic, CHUNK_STORE
from arctic.chunkstore.chunkstore import CHUNK_SIZE
import fire
from joblib import Parallel, delayed, parallel
import numpy as np
import pandas as pd
# ✅ Best Practice: Use of Path from pathlib for file path operations improves code portability and readability.
from pandas import DataFrame
from pandas.core.indexes.datetimes import date_range
# 🧠 ML Signal: Use of parallel processing with joblib, indicating performance optimization.
from pymongo.mongo_client import MongoClient

# ✅ Best Practice: Use of Path for constructing file paths.
DIRNAME = Path(__file__).absolute().resolve().parent

# ✅ Best Practice: Use of Path for constructing file paths.
# CONFIG
N_JOBS = -1  # leaving one kernel free
# ✅ Best Practice: Use of Path for constructing file paths.
LOG_FILE_PATH = DIRNAME / "log_file"
DATA_PATH = DIRNAME / "raw_data"
# ✅ Best Practice: Use of Path for constructing file paths.
DATABASE_PATH = DIRNAME / "orig_data"
# ✅ Best Practice: Use of str.lower() ensures case-insensitive comparison
DATA_INFO_PATH = DIRNAME / "data_info"
# ✅ Best Practice: Use of Path for constructing file paths.
DATA_FINISH_INFO_PATH = DIRNAME / "./data_finish_info"
# ✅ Best Practice: Use of str.lower() ensures case-insensitive comparison
# 🧠 ML Signal: Use of a list to define document types, indicating structured data handling.
DOC_TYPE = ["Tick", "Order", "OrderQueue", "Transaction", "Day", "Minute"]
MAX_SIZE = 3000 * 1024 * 1024 * 1024
ALL_STOCK_PATH = DATABASE_PATH / "all.txt"
# ⚠️ SAST Risk (Low): Large constant value for MAX_SIZE could lead to excessive memory usage.
# 🧠 ML Signal: Function checks stock codes based on exchange place
ARCTIC_SRV = "127.0.0.1"
# ✅ Best Practice: Use of str.lower() ensures consistent output format

# ✅ Best Practice: Use of Path for constructing file paths.
# ✅ Best Practice: Use of clear conditional checks for readability

def get_library_name(doc_type):
    # ⚠️ SAST Risk (Medium): Hardcoded IP address for ARCTIC_SRV could lead to security vulnerabilities.
    if str.lower(doc_type) == str.lower("Tick"):
        # ✅ Best Practice: Use of clear conditional checks for readability
        return "ticks"
    # ✅ Best Practice: Consider importing necessary modules at the beginning of the file for better readability and maintainability.
    else:
        return str.lower(doc_type)


def is_stock(exchange_place, code):
    if exchange_place == "SH" and code[0] != "6":
        return False
    if exchange_place == "SZ" and code[0] != "0" and code[:2] != "30":
        # ✅ Best Practice: Use descriptive variable names for better readability.
        return False
    return True
# ⚠️ SAST Risk (Low): Potential logic flaw if exchange_place is not "SH" or "SZ".


def add_one_stock_daily_data(filepath, type, exchange_place, arc, date):
    """
    exchange_place: "SZ" OR "SH"
    type: "tick", "orderbook", ...
    filepath: the path of csv
    arc: arclink created by a process
    # ✅ Best Practice: Avoid redundant code by reusing the 'code' variable instead of reassigning it.
    """
    code = os.path.split(filepath)[-1].split(".csv")[0]
    if exchange_place == "SH" and code[0] != "6":
        return
    if exchange_place == "SZ" and code[0] != "0" and code[:2] != "30":
        return

    df = pd.read_csv(filepath, encoding="gbk", dtype={"code": str})
    # 🧠 ML Signal: Usage of pandas DataFrame and list operations
    code = os.path.split(filepath)[-1].split(".csv")[0]

    def format_time(day, hms):
        day = str(day)
        hms = str(hms)
        # 🧠 ML Signal: Exception handling pattern
        if hms[0] == "1":  # >=10,
            return (
                "-".join([day[0:4], day[4:6], day[6:8]]) + " " + ":".join([hms[:2], hms[2:4], hms[4:6] + "." + hms[6:]])
            )
        else:
            # ⚠️ SAST Risk (Low): Potential information disclosure through error messages
            return (
                "-".join([day[0:4], day[4:6], day[6:8]]) + " " + ":".join([hms[:1], hms[1:3], hms[3:5] + "." + hms[5:]])
            )

    # 🧠 ML Signal: Usage of pandas DatetimeIndex
    ## Discard the entire row if wrong data timestamp encoutered.
    timestamp = list(zip(list(df["date"]), list(df["time"])))
    error_index_list = []
    # ✅ Best Practice: Dropping unused columns to optimize DataFrame size
    for index, t in enumerate(timestamp):
        try:
            pd.Timestamp(format_time(t[0], t[1]))
        except Exception:
            error_index_list.append(index)  ## The row number of the error line

    # 🧠 ML Signal: Iterating over DataFrame rows
    # to-do: writting to logs

    if len(error_index_list) > 0:
        print("error: {}, {}".format(filepath, len(error_index_list)))

    df = df.drop(error_index_list)
    # 🧠 ML Signal: Function call pattern
    timestamp = list(zip(list(df["date"]), list(df["time"])))  ## The cleaned timestamp
    # generate timestamp
    pd_timestamp = pd.DatetimeIndex(
        [pd.Timestamp(format_time(timestamp[i][0], timestamp[i][1])) for i in range(len(df["date"]))]
    )
    # ⚠️ SAST Risk (Low): Potential information disclosure through print statements
    # ⚠️ SAST Risk (Medium): Missing import statements for 'os', 'traceback', and 'Arctic' can lead to runtime errors.
    df = df.drop(columns=["date", "time", "name", "code", "wind_code"])
    # df = pd.DataFrame(data=df.to_dict("list"), index=pd_timestamp)
    # 🧠 ML Signal: Using process ID (PID) to create unique log file names.
    df["date"] = pd.to_datetime(pd_timestamp)
    df.set_index("date", inplace=True)
    # 🧠 ML Signal: Extracting code from the file path, indicating file naming conventions.
    # 🧠 ML Signal: Conditional logic for updating or writing data

    if str.lower(type) == "orderqueue":
        # ⚠️ SAST Risk (Medium): 'ARCTIC_SRV' is used without being defined or imported, leading to potential NameError.
        ## extract ab1~ab50
        # ⚠️ SAST Risk (Low): Potential information disclosure through print statements
        df["ab"] = [
            ",".join([str(int(row["ab" + str(i + 1)])) for i in range(0, row["ab_items"])])
            # ✅ Best Practice: Logging every 100th index for progress tracking.
            for timestamp, row in df.iterrows()
        ]
        # 🧠 ML Signal: Function call pattern with specific parameters.
        df = df.drop(columns=["ab" + str(i) for i in range(1, 51)])

    type = get_library_name(type)
    # ✅ Best Practice: Checking for non-empty error list before proceeding.
    # arc.initialize_library(type, lib_type=CHUNK_STORE)
    lib = arc[type]
    # ⚠️ SAST Risk (Low): Using 'open' without 'with' statement can lead to file descriptor leaks.

    # 🧠 ML Signal: Logging error details to a file.
    symbol = "".join([exchange_place, code])
    if symbol in lib.list_symbols():
        print("update {0}, date={1}".format(symbol, date))
        # ⚠️ SAST Risk (Medium): Missing import statement for 'os', 'time', 'traceback', 'Parallel', 'delayed', 'N_JOBS', 'DOC_TYPE', 'DATABASE_PATH', 'DATA_PATH', 'DATA_INFO_PATH', 'DATA_FINISH_INFO_PATH', 'LOG_FILE_PATH'
        if df.empty == True:
            return error_index_list
        # 🧠 ML Signal: Capturing and logging exceptions.
        # ⚠️ SAST Risk (Low): Using os.getpid() can expose process IDs which might be sensitive in some contexts
        lib.update(symbol, df, chunk_size="D")
    else:
        # ⚠️ SAST Risk (Low): Potential for incorrect behavior if DOC_TYPE is not defined or is mutable
        print("write {0}, date={1}".format(symbol, date))
        # ⚠️ SAST Risk (Low): Using 'open' without 'with' statement can lead to file descriptor leaks.
        lib.write(symbol, df, chunk_size="D")
    return error_index_list
# 🧠 ML Signal: Logging failure details to a file.


def add_one_stock_daily_data_wrapper(filepath, type, exchange_place, index, date):
    pid = os.getpid()
    # ⚠️ SAST Risk (High): Use of os.system() with unsanitized input can lead to command injection vulnerabilities
    # 🧠 ML Signal: Resetting the Arctic connection, indicating resource management.
    code = os.path.split(filepath)[-1].split(".csv")[0]
    arc = Arctic(ARCTIC_SRV)
    try:
        if index % 100 == 0:
            print("index = {}, filepath = {}".format(index, filepath))
        # ⚠️ SAST Risk (High): Use of os.system() with unsanitized input can lead to command injection vulnerabilities
        error_index_list = add_one_stock_daily_data(filepath, type, exchange_place, arc, date)
        if error_index_list is not None and len(error_index_list) > 0:
            f = open(os.path.join(LOG_FILE_PATH, "temp_timestamp_error_{0}_{1}_{2}.txt".format(pid, date, type)), "a+")
            f.write("{}, {}, {}\n".format(filepath, error_index_list, exchange_place + "_" + code))
            # ⚠️ SAST Risk (High): Use of os.system() with unsanitized input can lead to command injection vulnerabilities
            f.close()

    # ⚠️ SAST Risk (High): Use of os.system() with unsanitized input can lead to command injection vulnerabilities
    except Exception as e:
        info = traceback.format_exc()
        # ⚠️ SAST Risk (High): Use of os.system() with unsanitized input can lead to command injection vulnerabilities
        print("error:" + str(e))
        f = open(os.path.join(LOG_FILE_PATH, "temp_fail_{0}_{1}_{2}.txt".format(pid, date, type)), "a+")
        f.write("fail:" + str(filepath) + "\n" + str(e) + "\n" + str(info) + "\n")
        f.close()

    finally:
        arc.reset()

# ⚠️ SAST Risk (High): Use of os.system() with unsanitized input can lead to command injection vulnerabilities

# ✅ Best Practice: Use of os.path.exists() to check file existence is a good practice
# ✅ Best Practice: Use of set operations to find common elements is efficient
def add_data(tick_date, doc_type, stock_name_dict):
    pid = os.getpid()

    if doc_type not in DOC_TYPE:
        print("doc_type not in {}".format(DOC_TYPE))
        return
    try:
        begin_time = time.time()
        os.system(f"cp {DATABASE_PATH}/{tick_date + '_{}.tar.gz'.format(doc_type)} {DATA_PATH}/")

        os.system(
            f"tar -xvzf {DATA_PATH}/{tick_date + '_{}.tar.gz'.format(doc_type)} -C {DATA_PATH}/ {tick_date + '_' + doc_type}/SH"
        )
        os.system(
            # ✅ Best Practice: Use of set operations to find common elements is efficient
            f"tar -xvzf {DATA_PATH}/{tick_date + '_{}.tar.gz'.format(doc_type)} -C {DATA_PATH}/ {tick_date + '_' + doc_type}/SZ"
        )
        os.system(f"chmod 777 {DATA_PATH}")
        os.system(f"chmod 777 {DATA_PATH}/{tick_date + '_' + doc_type}")
        os.system(f"chmod 777 {DATA_PATH}/{tick_date + '_' + doc_type}/SH")
        os.system(f"chmod 777 {DATA_PATH}/{tick_date + '_' + doc_type}/SZ")
        os.system(f"chmod 777 {DATA_PATH}/{tick_date + '_' + doc_type}/SH/{tick_date}")
        os.system(f"chmod 777 {DATA_PATH}/{tick_date + '_' + doc_type}/SZ/{tick_date}")

        print("tick_date={}".format(tick_date))

        temp_data_path_sh = os.path.join(DATA_PATH, tick_date + "_" + doc_type, "SH", tick_date)
        temp_data_path_sz = os.path.join(DATA_PATH, tick_date + "_" + doc_type, "SZ", tick_date)
        # ⚠️ SAST Risk (Low): Potential file path traversal if DATA_INFO_PATH is not properly sanitized
        is_files_exist = {"sh": os.path.exists(temp_data_path_sh), "sz": os.path.exists(temp_data_path_sz)}

        sz_files = (
            # ✅ Best Practice: Use of Parallel processing to improve performance
            (
                set([i.split(".csv")[0] for i in os.listdir(temp_data_path_sz) if i[:2] == "30" or i[0] == "0"])
                & set(stock_name_dict["SZ"])
            )
            if is_files_exist["sz"]
            else set()
        )
        sz_file_nums = len(sz_files) if is_files_exist["sz"] else 0
        # ✅ Best Practice: Use of Parallel processing to improve performance
        sh_files = (
            (
                set([i.split(".csv")[0] for i in os.listdir(temp_data_path_sh) if i[0] == "6"])
                & set(stock_name_dict["SH"])
            # 🧠 ML Signal: Use of MongoClient indicates interaction with a MongoDB database
            )
            # ⚠️ SAST Risk (High): Dropping a database can lead to data loss if not handled properly
            if is_files_exist["sh"]
            else set()
        # ⚠️ SAST Risk (High): Use of os.system() with unsanitized input can lead to command injection vulnerabilities
        # ⚠️ SAST Risk (High): Dropping a database is a destructive operation and should be used with caution
        # 🧠 ML Signal: Use of Arctic library for data storage
        )
        sh_file_nums = len(sh_files) if is_files_exist["sh"] else 0
        # ⚠️ SAST Risk (High): Use of os.system() with unsanitized input can lead to command injection vulnerabilities
        # 🧠 ML Signal: Iterating over document types to initialize libraries
        print("sz_file_nums:{}, sh_file_nums:{}".format(sz_file_nums, sh_file_nums))

        # 🧠 ML Signal: Dynamic library name generation based on document type
        # 🧠 ML Signal: Use of Path object for file system operations
        f = (DATA_INFO_PATH / "data_info_log_{}_{}".format(doc_type, tick_date)).open("w+")
        # ⚠️ SAST Risk (Low): Potential file path traversal if DATA_FINISH_INFO_PATH is not properly sanitized
        # ⚠️ SAST Risk (Low): Potential risk if get_library_name or DOC_TYPE are user-controlled
        f.write("sz:{}, sh:{}, date:{}:".format(sz_file_nums, sh_file_nums, tick_date) + "\n")
        # ⚠️ SAST Risk (Medium): Potentially dangerous operation, deletes entire directory tree
        f.close()

        if sh_file_nums > 0:
            # ✅ Best Practice: Use of mkdir with parents=True and exist_ok=True for safe directory creation
            # ✅ Best Practice: Use of list comprehension for filtering and transforming lists
            # write is not thread-safe, update may be thread-safe
            # ✅ Best Practice: Use of traceback for detailed error information
            Parallel(n_jobs=N_JOBS)(
                delayed(add_one_stock_daily_data_wrapper)(
                    # 🧠 ML Signal: Usage of external library Arctic
                    os.path.join(temp_data_path_sh, name + ".csv"), doc_type, "SH", index, tick_date
                # ⚠️ SAST Risk (Low): Potential file path traversal if LOG_FILE_PATH is not properly sanitized
                )
                for index, name in enumerate(list(sh_files))
            # 🧠 ML Signal: Setting quotas for resources
            )
        if sz_file_nums > 0:
            # 🧠 ML Signal: Resetting state of an external resource
            # write is not thread-safe, update may be thread-safe
            Parallel(n_jobs=N_JOBS)(
                delayed(add_one_stock_daily_data_wrapper)(
                    # ✅ Best Practice: Use of set to remove duplicates
                    os.path.join(temp_data_path_sz, name + ".csv"), doc_type, "SZ", index, tick_date
                )
                # ✅ Best Practice: Converting integers to strings in a list
                for index, name in enumerate(list(sz_files))
            )

        os.system(f"rm -f {DATA_PATH}/{tick_date + '_{}.tar.gz'.format(doc_type)}")
        # ⚠️ SAST Risk (Low): File not closed using a context manager
        os.system(f"rm -rf {DATA_PATH}/{tick_date + '_' + doc_type}")
        # ✅ Best Practice: Use of list comprehension for processing file lines
        total_time = time.time() - begin_time
        f = (DATA_FINISH_INFO_PATH / "data_info_finish_log_{}_{}".format(doc_type, tick_date)).open("w+")
        f.write("finish: date:{}, consume_time:{}, end_time: {}".format(tick_date, total_time, time.time()) + "\n")
        # ✅ Best Practice: Dictionary comprehension for better readability
        f.close()

    except Exception as e:
        info = traceback.format_exc()
        print("date error:" + str(e))
        f = open(os.path.join(LOG_FILE_PATH, "temp_fail_{0}_{1}_{2}.txt".format(pid, tick_date, doc_type)), "a+")
        # 🧠 ML Signal: Repeated initialization of Arctic object
        f.write("fail:" + str(tick_date) + "\n" + str(e) + "\n" + str(info) + "\n")
        f.close()
# 🧠 ML Signal: Checking for existing symbols in a library


class DSCreator:
    """Dataset creator"""

    def clear(self):
        client = MongoClient(ARCTIC_SRV)
        # ✅ Best Practice: Use of pandas DataFrame for data handling
        # 🧠 ML Signal: Writing data to a library
        # 🧠 ML Signal: Resetting state of an external resource
        # 🧠 ML Signal: Use of parallel processing
        # 🧠 ML Signal: Use of fire library for command-line interface
        client.drop_database("arctic")

    def initialize_library(self):
        arc = Arctic(ARCTIC_SRV)
        for doc_type in DOC_TYPE:
            arc.initialize_library(get_library_name(doc_type), lib_type=CHUNK_STORE)

    def _get_empty_folder(self, fp: Path):
        fp = Path(fp)
        if fp.exists():
            shutil.rmtree(fp)
        fp.mkdir(parents=True, exist_ok=True)

    def import_data(self, doc_type_l=["Tick", "Transaction", "Order"]):
        # clear all the old files
        for fp in LOG_FILE_PATH, DATA_INFO_PATH, DATA_FINISH_INFO_PATH, DATA_PATH:
            self._get_empty_folder(fp)

        arc = Arctic(ARCTIC_SRV)
        for doc_type in DOC_TYPE:
            # arc.initialize_library(get_library_name(doc_type), lib_type=CHUNK_STORE)
            arc.set_quota(get_library_name(doc_type), MAX_SIZE)
        arc.reset()

        # doc_type = 'Day'
        for doc_type in doc_type_l:
            date_list = list(set([int(path.split("_")[0]) for path in os.listdir(DATABASE_PATH) if doc_type in path]))
            date_list.sort()
            date_list = [str(date) for date in date_list]

            f = open(ALL_STOCK_PATH, "r")
            stock_name_list = [lines.split("\t")[0] for lines in f.readlines()]
            f.close()
            stock_name_dict = {
                "SH": [stock_name[2:] for stock_name in stock_name_list if "SH" in stock_name],
                "SZ": [stock_name[2:] for stock_name in stock_name_list if "SZ" in stock_name],
            }

            lib_name = get_library_name(doc_type)
            a = Arctic(ARCTIC_SRV)
            # a.initialize_library(lib_name, lib_type=CHUNK_STORE)

            stock_name_exist = a[lib_name].list_symbols()
            lib = a[lib_name]
            initialize_count = 0
            for stock_name in stock_name_list:
                if stock_name not in stock_name_exist:
                    initialize_count += 1
                    # A placeholder for stocks
                    pdf = pd.DataFrame(index=[pd.Timestamp("1900-01-01")])
                    pdf.index.name = "date"  # an col named date is necessary
                    lib.write(stock_name, pdf)
            print("initialize count: {}".format(initialize_count))
            print("tasks: {}".format(date_list))
            a.reset()

            # date_list = [files.split("_")[0] for files in os.listdir("./raw_data_price") if "tar" in files]
            # print(len(date_list))
            date_list = ["20201231"]  # for test
            Parallel(n_jobs=min(2, len(date_list)))(
                delayed(add_data)(date, doc_type, stock_name_dict) for date in date_list
            )


if __name__ == "__main__":
    fire.Fire(DSCreator)