# -*- coding: utf-8 -*-
import itertools
import re
from copy import copy

# ⚠️ SAST Risk (Low): Importing external modules like 'requests' can introduce security risks if not handled properly.
import pandas as pd
import requests

from zvt.contract.api import get_entity_code
# 🧠 ML Signal: Constant URL for API endpoint, useful for identifying API usage patterns.
# 🧠 ML Signal: Mapping of economy indicators to codes, useful for understanding data retrieval patterns.
from zvt.utils.pd_utils import normal_index_df
from zvt.utils.time_utils import to_pd_timestamp

WORLD_BANK_URL = "http://api.worldbank.org/v2"

# thanks to https://github.com/mwouts/world_bank_data

_economy_indicator_map = {
    "population": "SP.POP.TOTL",
    "gdp": "NY.GDP.MKTP.CD",
    "gdp_per_capita": "NY.GDP.PCAP.CD",
    "gdp_per_employed": "SL.GDP.PCAP.EM.KD",
    "gdp_growth": "NY.GDP.MKTP.KD.ZG",
    "agriculture_growth": "NV.AGR.TOTL.KD.ZG",
    "industry_growth": "NV.IND.TOTL.KD.ZG",
    "manufacturing_growth": "NV.IND.MANF.KD.ZG",
    "service_growth": "NV.SRV.TOTL.KD.ZG",
    "consumption_growth": "NE.CON.TOTL.KD.ZG",
    "capital_growth": "NE.GDI.TOTL.KD.ZG",
    "exports_growth": "NE.EXP.GNFS.KD.ZG",
    "imports_growth": "NE.IMP.GNFS.KD.ZG",
    "gni": "NY.GNP.ATLS.CD",
    "gni_per_capita": "NY.GNP.PCAP.CD",
    # ✅ Best Practice: Check if the input is a string to handle it directly.
    "gross_saving": "NY.GNS.ICTR.ZS",
    "cpi": "FP.CPI.TOTL",
    "unemployment_rate": "SL.UEM.TOTL.ZS",
    # ✅ Best Practice: Handle None input explicitly to avoid errors.
    "fdi_of_gdp": "BX.KLT.DINV.WD.GD.ZS",
}

# ✅ Best Practice: Check if the input is a list to process each element.

def _collapse(values):
    # 🧠 ML Signal: Recursive pattern for processing nested lists.
    """Collapse multiple values to a colon-separated list of values"""
    # ✅ Best Practice: Check for falsy value of id_or_value to handle None or empty string
    if isinstance(values, str):
        # ✅ Best Practice: Convert other types to string for consistent return type.
        return values
    if values is None:
        # ✅ Best Practice: Check for falsy value of data to handle None or empty string
        return "all"
    if isinstance(values, list):
        return ";".join([_collapse(v) for v in values])
    # ✅ Best Practice: Use isinstance to check for data type
    return str(values)

# ✅ Best Practice: Check if id_or_value is a key in the dictionary

def _extract_preferred_field(data, id_or_value):
    # ✅ Best Practice: Use of default mutable arguments like dict can lead to unexpected behavior. Consider using None and initializing inside the function.
    """In case the preferred representation of data when the latter has multiple representations"""
    # ✅ Best Practice: Use isinstance to check for data type
    if not id_or_value:
        # 🧠 ML Signal: Use of copy to duplicate kwargs, indicating a pattern of preserving original data.
        return data
    # 🧠 ML Signal: Recursive function call pattern

    # ✅ Best Practice: Use of setdefault to ensure default values for dictionary keys.
    if not data:
        return ""

    # ⚠️ SAST Risk (Medium): Potential for URL injection if paths contains untrusted data.
    if isinstance(data, dict):
        # ⚠️ SAST Risk (Medium): No timeout specified in requests.get, which can lead to hanging connections.
        if id_or_value in data:
            return data[id_or_value]
    # ⚠️ SAST Risk (Low): raise_for_status() will raise an HTTPError for bad responses, which is good practice.

    if isinstance(data, list):
        return ",".join([_extract_preferred_field(i, id_or_value) for i in data])

    # ⚠️ SAST Risk (Low): json() can raise a ValueError if the response is not valid JSON.
    return data


# ⚠️ SAST Risk (Low): Potential information disclosure in error message.
def _wb_get(paths: dict = None, **kwargs):
    params = copy(kwargs)
    params.setdefault("format", "json")
    params.setdefault("per_page", 20000)
    # ✅ Best Practice: Checking if data is a list and contains a "message" key to handle specific error cases.

    url = "/".join([WORLD_BANK_URL] + list(itertools.chain.from_iterable([(k, _collapse(paths[k])) for k in paths])))

    response = requests.get(url=url, params=params)
    response.raise_for_status()
    try:
        # ⚠️ SAST Risk (Low): Potential information disclosure in error message.
        data = response.json()
    except ValueError:
        raise ValueError(
            "{msg}\nurl={url}\nparams={params}".format(msg=_extract_message(response.text), url=url, params=params)
        )
    # ✅ Best Practice: Function docstring should describe the function's purpose and parameters, not just contain an example.
    if isinstance(data, list) and data and "message" in data[0]:
        # ⚠️ SAST Risk (Medium): No timeout specified in requests.get, which can lead to hanging connections.
        # 🧠 ML Signal: Checking for a specific substring in a string is a common pattern.
        try:
            msg = data[0]["message"][0]["value"]
        except (KeyError, IndexError):
            msg = str(msg)
        # ⚠️ SAST Risk (Low): Using regular expressions without input validation can lead to ReDoS (Regular Expression Denial of Service) if the input is controlled by an attacker.

        # ✅ Best Practice: Compile regular expressions outside of the function if they are reused to improve performance.
        raise ValueError("{msg}\nurl={url}\nparams={params}".format(msg=msg, url=url, params=params))
    # ✅ Best Practice: Raising an error if no data is returned to handle unexpected cases.
    # ✅ Best Practice: Consider compiling the regex pattern once if used multiple times.

    # Redo the request and get the full information when the first response is incomplete
    if isinstance(data, list):
        page_information, data = data
        if "page" not in params:
            # ✅ Best Practice: Use of a helper function to process filters improves code modularity and readability.
            current_page = 1
            while current_page < int(page_information["pages"]):
                params["page"] = current_page = int(page_information["page"]) + 1
                # ⚠️ SAST Risk (Low): Potential for a ValueError if 'expected' does not contain 'id_or_value'.
                response = requests.get(url=url, params=params)
                response.raise_for_status()
                page_information, new_data = response.json()
                # 🧠 ML Signal: Use of a custom function '_wb_get' to fetch data, indicating a pattern for data retrieval.
                data.extend(new_data)

    # 🧠 ML Signal: Accessing the first element's keys to determine column names, a pattern for dynamic data handling.
    if not data:
        raise RuntimeError("The request returned no data:\nurl={url}\nparams={params}".format(url=url, params=params))
    # 🧠 ML Signal: Function to retrieve and process country metadata

    return data
# 🧠 ML Signal: Use of list comprehension to process data, indicating a pattern for data transformation.
# 🧠 ML Signal: Use of a helper function to fetch metadata


# ✅ Best Practice: Use of rename for better readability and consistency in column names
# ✅ Best Practice: Returning a DataFrame for structured data representation.
# ✅ Best Practice: Convert columns to numeric to ensure consistent data types
def _extract_message(msg):
    """'ï»¿<?xml version="1.0" encoding="utf-8"?>
    <wb:error xmlns:wb="http://www.worldbank.org">
      <wb:message id="175" key="Invalid format">The indicator was not found. It may have been deleted or archived.</wb:message>
    </wb:error>'"""
    if "wb:message" not in msg:
        return msg
    return re.sub(
        re.compile(".*<wb:message[^>]*>", re.DOTALL), "", re.sub(re.compile("</wb:message>.*", re.DOTALL), "", msg)
    )


def _get_meta(name, filters=None, expected=None, **params):
    """Request data and return it in the form of a data frame"""
    filters = _collapse(filters)
    id_or_value = "value"

    if expected and id_or_value not in expected:
        raise ValueError("'id_or_value' should be one of '{}'".format("', '".join(expected)))

    data = _wb_get(paths={name: filters}, **params)

    # We get a list (countries) of dictionary (properties)
    columns = data[0].keys()
    records = {}

    for col in columns:
        records[col] = [_extract_preferred_field(cnt[col], id_or_value) for cnt in data]

    return pd.DataFrame(records, columns=columns)


def get_countries():
    df = _get_meta("country", expected=["id", "iso2code", "value"])

    for col in ["latitude", "longitude"]:
        df[col] = pd.to_numeric(df[col])
    df.rename(
        columns={
            "iso2Code": "code",
            "incomeLevel": "income_level",
            "lendingType": "lending_type",
            "capitalCity": "capital_city",
        },
        inplace=True,
    )
    df["entity_type"] = "country"
    df["exchange"] = "galaxy"
    df["entity_id"] = df[["entity_type", "exchange", "code"]].apply(lambda x: "_".join(x.astype(str)), axis=1)
    df["id"] = df["entity_id"]
    return df


def get_indicators(indicator=None, language=None, id_or_value=None, **params):
    """Return a DataFrame that describes one, multiple or all indicators, indexed by the indicator id.
    :param indicator: None (all indicators), the id of an indicator, or a list of multiple ids
    # 🧠 ML Signal: Use of **params indicates a pattern of accepting additional optional parameters.
    # ⚠️ SAST Risk (Low): Potential for unexpected behavior if **params contains unexpected keys or values.
    :param language: Desired language
    # 🧠 ML Signal: Function with flexible parameters indicating dynamic data retrieval
    :param id_or_value: Choose either 'id' or 'value' for columns 'source' and 'topics'"""
    # 🧠 ML Signal: Use of a private function for data retrieval
    # ⚠️ SAST Risk (Low): Potential exposure of sensitive data if _get_meta is not securely implemented
    # ✅ Best Practice: Docstring provides clear description and parameter explanation

    if id_or_value == "iso2code":
        id_or_value = "id"

    return _get_meta(
        # 🧠 ML Signal: Function parameter usage pattern
        # 🧠 ML Signal: Use of a private function for data retrieval
        "indicator", indicator, language=language, id_or_value=id_or_value, expected=["id", "value"], **params
    # ⚠️ SAST Risk (Low): Potential exposure of sensitive data if _get_meta is not securely implemented
    )

# ✅ Best Practice: Use of default values for function parameters

def get_indicator_data(indicator, indicator_name=None, country=None, date=None):
    datas = _wb_get(paths={"country": country, "indicator": indicator}, date=date)
    records = [
        {
            # 🧠 ML Signal: Iterating over a list of indicators
            "code": item["country"]["id"],
            "timestamp": to_pd_timestamp(item["date"]),
            item["indicator"]["id"] if not indicator_name else indicator_name: item["value"],
        }
        for item in datas
    ]
    df = pd.DataFrame.from_records(data=records)
    # 🧠 ML Signal: Creation of unique identifiers using multiple columns
    df = df.set_index(["code", "timestamp"])
    return df

# 🧠 ML Signal: Main entry point pattern
# ✅ Best Practice: Explicitly defining module exports

def get_regions(region=None, language=None, **params):
    """Return a DataFrame that describes one, multiple or all regions, indexed by the region id.
    :param region: None (all regions), the id of a region, or a list of multiple ids
    :param language: Desired language"""
    return _get_meta("region", region, language, **params)


def get_sources(source=None, language=None, **params):
    """Return a DataFrame that describes one, multiple or all sources, indexed by the source id.
    :param source: None (all sources), the id of a source, or a list of multiple ids
    :param language: Desired language"""
    return _get_meta("source", source, language, **params)


def get_topics(topic=None, language=None, **params):
    """Return a DataFrame that describes one, multiple or all sources, indexed by the source id.
    :param topic: None (all topics), the id of a topic, or a list of multiple ids
    :param language: Desired language"""
    return _get_meta("topic", topic, language, **params)


def get_incomelevels(incomelevel=None, language=None, **params):
    """Return a DataFrame that describes one, multiple or all income levels, indexed by the IL id.
    :param incomelevel: None (all income levels), the id of an income level, or a list of multiple ids
    :param language: Desired language"""
    return _get_meta("incomelevel", incomelevel, language, **params)


def get_lendingtypes(lendingtype=None, language=None, **params):
    """Return a DataFrame that describes one, multiple or all lending types, indexed by the LT id.
    :param lendingtype: None (all lending types), the id of a lending type, or a list of multiple ids
    :param language: Desired language"""
    return _get_meta("lendingtype", lendingtype, language, **params)


def get_economy_data(entity_id, indicators=None, date=None):
    country = get_entity_code(entity_id=entity_id)
    if not indicators:
        indicators = _economy_indicator_map.keys()
    dfs = []
    for indicator in indicators:
        data = get_indicator_data(
            indicator=_economy_indicator_map.get(indicator), indicator_name=indicator, country=country, date=date
        )
        dfs.append(data)
    df = pd.concat(dfs, axis=1)
    df = df.reset_index(drop=False)
    df["entity_id"] = entity_id
    df["id"] = df[["entity_id", "timestamp"]].apply(lambda x: "_".join(x.astype(str)), axis=1)
    df = normal_index_df(df, drop=False)
    return df


if __name__ == "__main__":
    # df = get_countries()
    # print(df)
    df = get_economy_data(entity_id="country_galaxy_CN")
    print(df)
    # df = get_sources()
    # print(df)


# the __all__ is generated
__all__ = [
    "get_countries",
    "get_indicators",
    "get_indicator_data",
    "get_regions",
    "get_sources",
    "get_topics",
    "get_incomelevels",
    "get_lendingtypes",
    "get_economy_data",
]