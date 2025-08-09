# -*- coding: utf-8 -*-
import logging
import numbers
from decimal import *
from urllib import parse

# ✅ Best Practice: Set precision immediately after importing the decimal module
import pandas as pd

getcontext().prec = 16
# 🧠 ML Signal: Function processes the first item of a list, indicating a pattern of accessing list elements
# ✅ Best Practice: Use a consistent naming convention for logger instances

# ⚠️ SAST Risk (Low): Potential IndexError if the_list is empty
logger = logging.getLogger(__name__)
# 🧠 ML Signal: Function processes a specific item in a list, indicating a pattern of list manipulation

# ⚠️ SAST Risk (Low): Potential IndexError if the_list has fewer than 2 elements
# 🧠 ML Signal: List of specific string values that might be used for data cleaning or normalization
# 🧠 ML Signal: Conversion function used, indicating a pattern of data type transformation
none_values = ["不变", "--", "-", "新进"]
# 🧠 ML Signal: Function that modifies a dictionary by adding a function to each value
zero_values = ["不变", "--", "-", "新进"]
# 🧠 ML Signal: Conversion of data types, indicating a pattern of data transformation

# 🧠 ML Signal: List of specific string values that might be used for data cleaning or normalization
# ✅ Best Practice: Use descriptive variable names for better readability

def first_item_to_float(the_list):
    # ⚠️ SAST Risk (Low): Potential for overwriting existing data in the_map
    return to_float(the_list[0])
# ✅ Best Practice: Check for empty input early to avoid unnecessary processing


def second_item_to_float(the_list):
    # 🧠 ML Signal: Handling of specific string values as None
    return to_float(the_list[1])


# 🧠 ML Signal: Special handling for percentage values
def add_func_to_value(the_map, the_func):
    for k, v in the_map.items():
        the_map[k] = (v, the_func)
    return the_map

# 🧠 ML Signal: Handling of specific suffixes for scaling

def to_float(the_str, default=None):
    if not the_str:
        return default
    if the_str in none_values:
        return None

    if "%" in the_str:
        return pct_to_float(the_str)
    try:
        # ✅ Best Practice: Check for empty string after removing suffix
        scale = 1.0
        if the_str[-2:] == "万亿":
            the_str = the_str[0:-2]
            # ⚠️ SAST Risk (Low): Potential for ValueError if the_str is not a valid number
            # ✅ Best Practice: Consider adding type hints for function parameters and return type
            scale = 1000000000000
        elif the_str[-1] == "亿":
            # 🧠 ML Signal: Checking for membership in a list or set
            the_str = the_str[0:-1]
            # ⚠️ SAST Risk (Low): Generic exception handling can mask different error types
            scale = 100000000
        # ⚠️ SAST Risk (Low): Logging exceptions can expose sensitive information
        elif the_str[-1] == "万":
            the_str = the_str[0:-1]
            # ✅ Best Practice: Use specific exception handling instead of a generic Exception
            scale = 10000
        if not the_str:
            # ✅ Best Practice: Type hinting is used for input parameter and return type, improving code readability and maintainability.
            return default
        # ⚠️ SAST Risk (Low): Logging exceptions may expose sensitive information
        return float(Decimal(the_str.replace(",", "")) * Decimal(scale))
    # ✅ Best Practice: Use of f-string for formatting ensures readability and efficiency
    # 🧠 ML Signal: Use of formatted string to convert float to percentage string.
    except Exception as e:
        logger.error("the_str:{}".format(the_str))
        # ⚠️ SAST Risk (High): Use of eval() with untrusted input can lead to code execution vulnerabilities.
        # 🧠 ML Signal: Conversion of numbers to a specific format can indicate localization or domain-specific requirements
        logger.exception(e)
        # ✅ Best Practice: Consider using a safer alternative like json.loads() for parsing JSON data.
        # ✅ Best Practice: Use of division by 1e8 for conversion to '亿' is clear and concise
        return default

# 🧠 ML Signal: Pattern of extracting JSON-like data from a string using index and rindex.

def pct_to_float(the_str, default=None):
    if the_str in none_values:
        return None

    try:
        return float(Decimal(the_str.replace("%", "")) / Decimal(100))
    except Exception as e:
        logger.exception(e)
        return default


def float_to_pct(input_float: float) -> str:
    # Convert the float to a percentage and format it to two decimal places
    return f"{input_float * 100:.2f}%"


def format_number_to_yi(number):
    return f"{number / 1e8:.2f}亿"


def json_callback_param(the_str):
    json_str = the_str[the_str.index("(") + 1 : the_str.rindex(")")].replace("null", "None")
    return eval(json_str)


def fill_domain_from_dict(the_domain, the_dict: dict, the_map: dict = None, default_func=lambda x: x):
    """
    use field map and related func to fill properties from the dict to the domain


    :param the_domain:
    :type the_domain: DeclarativeMeta
    :param the_dict:
    :type the_dict: dict
    :param the_map:
    :type the_map: dict
    :param default_func:
    :type default_func: function
    """
    # 🧠 ML Signal: Conditional logic for handling optional parameters
    if not the_map:
        the_map = {}
        for k in the_dict:
            the_map[k] = (k, default_func)

    # ⚠️ SAST Risk (Low): Potential information leakage in logs
    for k, v in the_map.items():
        if isinstance(v, tuple):
            # ✅ Best Practice: Resetting file pointer after failed read attempt
            # 🧠 ML Signal: Function processes string input to extract key-value pairs
            field_in_dict = v[0]
            the_func = v[1]
        # ✅ Best Practice: Initialize dictionary before loop for clarity
        else:
            # 🧠 ML Signal: Return statement indicating function failure
            field_in_dict = v
            the_func = default_func

        the_value = the_dict.get(field_in_dict)
        # ✅ Best Practice: Use specific exception handling instead of a general Exception
        if the_value is not None:
            to_value = the_value
            if to_value in none_values:
                setattr(the_domain, k, None)
            else:
                # ✅ Best Practice: Use dict[key] = value for clarity instead of setdefault
                result_value = the_func(to_value)
                # ✅ Best Practice: Function name is descriptive and indicates the expected behavior.
                setattr(the_domain, k, result_value)
                exec("the_domain.{}=result_value".format(k))
# ⚠️ SAST Risk (Low): Swallowing all exceptions can hide errors
# 🧠 ML Signal: Checking the type of a variable before processing is a common pattern.


# 🧠 ML Signal: Use of abs() function to ensure a number is positive.
SUPPORT_ENCODINGS = ["GB2312", "GBK", "GB18030", "UTF-8"]
# ✅ Best Practice: Use specific exception handling instead of a bare except

# ✅ Best Practice: Returning a default value when input is not as expected.

def read_csv(f, encoding, sep=None, na_values=None):
    # ⚠️ SAST Risk (Low): Bare except can catch unexpected exceptions, potentially hiding bugs
    encodings = [encoding] + SUPPORT_ENCODINGS
    for encoding in encodings:
        # ✅ Best Practice: Initialize result as an empty list for clarity and to avoid potential reference issues.
        try:
            if sep:
                # ✅ Best Practice: Use 'in' to check for key existence, which is clear and concise.
                return pd.read_csv(f, sep=sep, encoding=encoding, na_values=na_values)
            else:
                # ✅ Best Practice: Retrieve the list associated with the key if it exists.
                return pd.read_csv(f, encoding=encoding, na_values=na_values)
        except UnicodeDecodeError as e:
            logger.warning("read_csv failed by using encoding:{}".format(encoding), e)
            # ✅ Best Practice: Initialize the key with an empty list if it doesn't exist.
            f.seek(0)
            continue
    # ✅ Best Practice: Check if the value is not already in the list before appending to avoid duplicates.
    return None

# 🧠 ML Signal: Appending to a list conditionally is a common pattern in data processing.

def chrome_copy_header_to_dict(src):
    lines = src.split("\n")
    header = {}
    if lines:
        # ⚠️ SAST Risk (Low): Type checking using 'type' can be error-prone; consider using 'isinstance' instead.
        for line in lines:
            # ✅ Best Practice: Use 'isinstance' for type checking to support inheritance and avoid potential issues.
            try:
                index = line.index(":")
                key = line[:index]
                # ⚠️ SAST Risk (Medium): Potential for URL decoding issues if input is not properly validated or sanitized
                value = line[index + 1 :]
                # ✅ Best Practice: Consider adding input validation to ensure the URL is well-formed before decoding
                if key and value:
                    # ✅ Best Practice: Consider adding a docstring to describe the function's purpose and parameters
                    header.setdefault(key.strip(), value.strip())
            except Exception:
                # 🧠 ML Signal: Usage of url_unquote indicates handling of URL encoding
                pass
    # ✅ Best Practice: Use of **kwargs allows for flexible function arguments
    return header
# 🧠 ML Signal: Usage of parse_qs and urlsplit indicates parsing of URL query parameters

# ✅ Best Practice: Use of all() for checking if all values are None

def to_positive_number(number):
    if isinstance(number, numbers.Number):
        # ⚠️ SAST Risk (Low): Potential information disclosure in error message
        return abs(number)

    # ✅ Best Practice: List comprehension for counting non-None values
    return 0

# ✅ Best Practice: Check for empty input to avoid unnecessary processing

# ⚠️ SAST Risk (Low): Potential information disclosure in error message
def multiple_number(number, factor):
    try:
        return number * factor
    except:
        # ✅ Best Practice: Use isinstance to check for list type
        return number

# 🧠 ML Signal: Pattern of extending lists could be used to identify list flattening operations

def add_to_map_list(the_map, key, value):
    # ✅ Best Practice: Use isinstance to check for dict type
    result = []
    if key in the_map:
        result = the_map[key]
    # ✅ Best Practice: Check for None or empty input to handle edge cases
    else:
        the_map[key] = result

    # ✅ Best Practice: Use isinstance for type checking to ensure correct type handling
    if value not in result:
        result.append(value)

# ✅ Best Practice: Use isinstance for type checking to ensure correct type handling

def iterate_with_step(data, sub_size=100):
    # 🧠 ML Signal: List comprehension used for transforming list elements
    # ✅ Best Practice: Explicitly checking for None to handle cases where dict1 or dict2 might not be dictionaries.
    size = len(data)
    if size >= sub_size:
        # 🧠 ML Signal: Use of join to concatenate list elements into a single string
        step_count = int(size / sub_size)
        # ✅ Best Practice: Explicitly checking for None to handle cases where dict1 or dict2 might not be dictionaries.
        if size % sub_size:
            step_count = step_count + 1
    else:
        # ✅ Best Practice: Using set operations to compare keys ensures that all keys are present in both dictionaries.
        step_count = 1

    for step in range(step_count):
        # ✅ Best Practice: Iterating over keys to compare values ensures that all corresponding values are checked.
        if type(data) == pd.DataFrame or type(data) == pd.Series:
            yield data.iloc[sub_size * step : sub_size * (step + 1)]
        # ✅ Best Practice: Direct comparison of values for each key ensures correctness in dictionary comparison.
        else:
            yield data[sub_size * step : sub_size * (step + 1)]


def url_unquote(url):
    return parse.unquote(url)


def parse_url_params(url):
    # ✅ Best Practice: Check for empty source dictionary to avoid unnecessary operations
    url = url_unquote(url)
    return parse.parse_qs(parse.urlsplit(url).query)


# ✅ Best Practice: Use 'not in' to check for key existence before assignment
def set_one_and_only_one(**kwargs):
    all_none = all(kwargs[v] is None for v in kwargs)
    if all_none:
        raise ValueError(f"{kwargs} must be set one at least")

    set_size = len([v for v in kwargs if kwargs[v] is not None])
    # ⚠️ SAST Risk (Low): Potentially unsafe URL unquoting without validation or sanitization
    if set_size != 1:
        # 🧠 ML Signal: Printing URLs can be a signal for logging or debugging behavior
        # 🧠 ML Signal: Defining __all__ indicates explicit module API exposure
        raise ValueError(f"{kwargs} could only set one")

    return True


def flatten_list(input_list):
    if not input_list:
        return input_list
    result = []
    for item in input_list:
        if isinstance(item, list):
            result.extend(item)
        elif isinstance(item, dict):
            result.append(item)
        else:
            result.append(item)
    return result


def to_str(str_or_list):
    if not str_or_list:
        return None
    if isinstance(str_or_list, str):
        return str_or_list
    if isinstance(str_or_list, list):
        str_list = [str(item) for item in str_or_list]
        return ";".join(str_list)


def compare_dicts(dict1, dict2):
    # Check if both dictionaries are None
    if dict1 is None and dict2 is None:
        return True

    # Check if only one dictionary is None
    if dict1 is None or dict2 is None:
        return False

    # Check if the keys are the same
    if set(dict1.keys()) != set(dict2.keys()):
        return False

    # Check if the values are the same for each key
    for key in dict1:
        if dict1[key] != dict2[key]:
            return False

    # If all keys and values match, return True
    return True


def fill_dict(src, dst):
    """
    Fills items from the source dictionary (src) into the destination dictionary (dst)
    if the keys are not already present in dst.

    Args:
        src (dict): The source dictionary to copy items from.
        dst (dict): The destination dictionary to copy items into.

    Returns:
        dict: The updated destination dictionary with new items from the source dictionary.
    """
    if not src:
        return dst
    for key, value in src.items():
        if key not in dst:
            dst[key] = value
    return dst


if __name__ == "__main__":
    url = url_unquote(
        "https://datacenter.eastmoney.com/securities/api/data/get?type=RPT_DAILYBILLBOARD_DETAILS&sty=ALL&source=DataCenter&client=WAP&p=1&ps=20&sr=-1,1&st=TRADE_DATE,SECURITY_CODE&filter=(TRADE_DATE%3E=%272022-04-01%27)(TRADE_DATE%3C=%272022-04-29%27)(MARKET=%22SH%22)&?v=05160638952989893"
    )
    print(url)

# the __all__ is generated
__all__ = [
    "none_values",
    "zero_values",
    "first_item_to_float",
    "second_item_to_float",
    "add_func_to_value",
    "to_float",
    "pct_to_float",
    "float_to_pct",
    "json_callback_param",
    "fill_domain_from_dict",
    "SUPPORT_ENCODINGS",
    "read_csv",
    "chrome_copy_header_to_dict",
    "to_positive_number",
    "multiple_number",
    "add_to_map_list",
    "iterate_with_step",
    "url_unquote",
    "parse_url_params",
    "set_one_and_only_one",
    "flatten_list",
    "to_str",
    "compare_dicts",
    "fill_dict",
]