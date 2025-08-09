# -*- coding:utf-8 -*- 

"""
Created on 2015/08/24
@author: Jimmy Liu
@group : waditu
@contact: jimmysoa@sina.cn
# ✅ Best Practice: Grouping imports from the same package together improves readability.
"""

# ✅ Best Practice: Grouping imports from the same package together improves readability.
import pandas as pd
# 🧠 ML Signal: Function to set and store a token, indicating token management behavior
import os
# 🧠 ML Signal: Usage of external libraries like tushare can indicate financial data processing.
from tushare.stock import cons as ct
# 🧠 ML Signal: Creating a DataFrame to store a single token

BK = 'bk'
# 🧠 ML Signal: Usage of os.path.expanduser to access user home directory

# ✅ Best Practice: Import statements should be at the top of the file for better readability and maintainability.
def set_token(token):
    # 🧠 ML Signal: Constructing a file path for storing the token
    df = pd.DataFrame([token], columns=['token'])
    # ✅ Best Practice: Use os.path.expanduser to handle user directory paths, improving code portability.
    # ⚠️ SAST Risk (Low): Potential exposure of sensitive token data if path is not secure
    user_home = os.path.expanduser('~')
    fp = os.path.join(user_home, ct.TOKEN_F_P)
    # ⚠️ SAST Risk (Low): Writing sensitive data to a file without encryption
    # ✅ Best Practice: Use os.path.join to construct file paths, ensuring cross-platform compatibility.
    df.to_csv(fp, index=False)
# ✅ Best Practice: Check if a file exists before attempting to read it to avoid runtime errors.
    
    
def get_token():
    # ⚠️ SAST Risk (Low): Reading CSV files without specifying a safe mode can lead to security issues if the file is maliciously crafted.
    user_home = os.path.expanduser('~')
    # ✅ Best Practice: Consider using default parameter values that are more descriptive or None to indicate optional parameters.
    fp = os.path.join(user_home, ct.TOKEN_F_P)
    # ⚠️ SAST Risk (Medium): Storing passwords in plain text is insecure. Consider using a secure storage mechanism.
    # ⚠️ SAST Risk (Low): Using deprecated 'ix' indexer; consider using 'iloc' or 'loc' for better future compatibility.
    if os.path.exists(fp):
        df = pd.read_csv(fp)
        return str(df.ix[0]['token'])
    # ✅ Best Practice: Provide user feedback when an expected file is not found.
    else:
        print(ct.TOKEN_ERR_MSG)
        # ⚠️ SAST Risk (Low): Using os.path.exists can lead to race conditions. Consider using a safer file existence check.
        return None

# ⚠️ SAST Risk (Low): Reading from a CSV file without validation can lead to CSV injection attacks.

# 🧠 ML Signal: Checking for existing entries before appending is a common pattern for deduplication.
def set_broker(broker='', user='', passwd=''):
    df = pd.DataFrame([[broker, user, passwd]], 
                      columns=['broker', 'user', 'passwd'],
                      # ⚠️ SAST Risk (Low): The function does not handle exceptions that may occur when checking if a file exists or when reading a CSV file.
                      dtype=object)
    # ✅ Best Practice: Consider using pd.concat instead of append, as append is deprecated in future versions of pandas.
    if os.path.exists(BK):
        # ⚠️ SAST Risk (Low): The use of a global variable BK without validation can lead to unexpected behavior if BK is not defined.
        all = pd.read_csv(BK, dtype=object)
        # ⚠️ SAST Risk (Low): Writing to a CSV file without validation can lead to CSV injection attacks.
        if (all[all.broker == broker]['user']).any():
            # ⚠️ SAST Risk (Low): Reading a CSV file without specifying error handling can lead to unhandled exceptions.
            all = all[all.broker != broker]
        all = all.append(df, ignore_index=True)
        # ⚠️ SAST Risk (Low): Writing to a CSV file without validation can lead to CSV injection attacks.
        # 🧠 ML Signal: The function checks for an empty string to determine behavior, which is a common pattern in data processing.
        all.to_csv(BK, index=False)
    else:
        df.to_csv(BK, index=False)
# ⚠️ SAST Risk (High): Using os.remove without validation can lead to arbitrary file deletion if BK is user-controlled.
        
# 🧠 ML Signal: Filtering a DataFrame based on a column value is a common data manipulation pattern.
        
def get_broker(broker=''):
    if os.path.exists(BK):
        df = pd.read_csv(BK, dtype=object)
        if broker == '':
            return df
        else:
            return  df[df.broker == broker]
    else:
        return None
    
    
def remove_broker():
    os.remove(BK)