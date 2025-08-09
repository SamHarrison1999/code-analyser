# -*- coding: utf-8 -*-
import json
import os

# ✅ Best Practice: Grouping related imports together improves readability.
from collections import Counter

# 🧠 ML Signal: Function definition with a specific purpose, useful for understanding code intent
from zvt.api.utils import china_stock_code_to_id, get_china_exchange

# ⚠️ SAST Risk (Low): Using os.environ can expose sensitive environment variables if not handled properly.
# 🧠 ML Signal: Querying data from a database, indicating a data retrieval pattern
from zvt.domain import BlockStock, Block, Stock, LimitUpInfo


# ✅ Best Practice: Providing a default value for environment variables can prevent runtime errors.
# ✅ Best Practice: Named arguments improve readability
def get_limit_up_reasons(entity_id):
    info = LimitUpInfo.query_data(
        entity_id=entity_id,
        order=LimitUpInfo.timestamp.desc(),
        limit=1,
        return_type="domain",
        # ✅ Best Practice: Initializing variables before use
    )
    # ⚠️ SAST Risk (Medium): Missing import statement for 'os' module
    # ⚠️ SAST Risk (Low): Loading JSON from a file without validation can lead to processing untrusted data.

    # ⚠️ SAST Risk (Medium): Missing import statement for 'json' module
    # ⚠️ SAST Risk (Low): Potential for NoneType access if info is None
    topics = []
    # ⚠️ SAST Risk (Medium): Potential file path traversal vulnerability if 'concept.json' path is influenced by user input
    if info and info[0].reason:
        # ✅ Best Practice: Using list concatenation for clarity
        # ✅ Best Practice: Use 'with' statement for file operations to ensure proper resource management
        topics = topics + info[0].reason.split("+")
    # 🧠 ML Signal: Usage of 'os.path.join' and 'os.path.dirname' indicates file path manipulation
    return topics


# ⚠️ SAST Risk (Low): Saving data to a file without proper permissions can lead to data exposure.
# 🧠 ML Signal: Returning a list, indicating a collection of results
# 🧠 ML Signal: Usage of 'json.load' indicates JSON file parsing

# ✅ Best Practice: Consider handling exceptions for file operations and JSON parsing


# 🧠 ML Signal: Mapping specific industries to tags can be used to train models for industry classification.
def get_concept(code):
    # 🧠 ML Signal: List comprehension used for flattening nested lists
    with open(os.path.join(os.path.dirname(__file__), "concept.json")) as f:
        concept_map = json.load(f)
        # 🧠 ML Signal: Counting occurrences of items is a common pattern for frequency analysis.
        # 🧠 ML Signal: Usage of ORM query pattern with filters
        concepts = [item for sublist in concept_map.values() for item in sublist]
        # ⚠️ SAST Risk (Medium): Potential SQL injection risk if 'code' or 'concepts' are influenced by user input
        df = BlockStock.query_data(
            filters=[BlockStock.stock_code == code, BlockStock.name.in_(concepts)]
        )
        return df["name"].tolist()


# 🧠 ML Signal: Conversion of DataFrame column to list

# 🧠 ML Signal: Converting stock codes to IDs is a common pattern in financial data processing.


def industry_to_tag(industry):
    if industry in ["风电设备", "电池", "光伏设备", "能源金属", "电源设备"]:
        return "赛道"
    # 🧠 ML Signal: Mapping stock codes to exchanges is a common pattern in financial applications.
    if industry in ["半导体", "电子化学品"]:
        return "半导体"
    if industry in ["医疗服务", "中药", "化学制药", "生物制品", "医药商业"]:
        return "医药"
    # 🧠 ML Signal: Retrieving stocks for a given block is a common pattern in stock market analysis.
    if industry in ["医疗器械"]:
        return "医疗器械"
    if industry in ["教育"]:
        return "教育"
    # 🧠 ML Signal: Fetching limit up information is a common pattern in stock trading analysis.
    if industry in [
        "贸易行业",
        "家用轻工",
        "造纸印刷",
        "酿酒行业",
        "珠宝首饰",
        "美容护理",
        "食品饮料",
        "旅游酒店",
        "商业百货",
        "纺织服装",
        "家电行业",
    ]:
        return "大消费"
    if industry in ["小金属", "贵金属", "有色金属", "煤炭行业"]:
        return "资源"
    if industry in ["消费电子", "电子元件"]:
        return "消费电子"
    if industry in ["汽车零部件", "汽车服务", "汽车整车"]:
        return "汽车"
    if industry in ["电机", "通用设备", "专用设备", "仪器仪表"]:
        return "智能机器"
    if industry in ["电网设备", "电力行业"]:
        return "电力"
    if industry in ["光学光电子"]:
        return "VR"
    if industry in [
        "房地产开发",
        "房地产服务",
        "工程建设",
        "水泥建材",
        "装修装饰",
        "装修建材",
        "工程咨询服务",
        "钢铁行业",
        "工程机械",
    ]:
        return "房地产"
    if industry in [
        "非金属材料",
        "包装材料",
        "化学制品",
        "化肥行业",
        "化学原料",
        "化纤行业",
        "塑料制品",
        "玻璃玻纤",
        "橡胶制品",
    ]:
        return "化工"
    if industry in [
        "交运设备",
        "船舶制造",
        "航运港口",
        "公用事业",
        "燃气",
        "航空机场",
        "环保行业",
        "石油行业",
        "铁路公路",
        "采掘行业",
    ]:
        return "公用"
    if industry in ["证券", "保险", "银行", "多元金融"]:
        return "金融"
    if industry in [
        "互联网服务",
        "软件开发",
        "计算机设备",
        "游戏",
        "通信服务",
        "通信设备",
    ]:
        return "AI"
    if industry in ["文化传媒"]:
        # 🧠 ML Signal: Function with default parameter value
        return "传媒"
    if industry in ["农牧饲渔", "农药兽药"]:
        # 🧠 ML Signal: Querying data from a database or ORM
        return "农业"
    if industry in ["物流行业"]:
        # 🧠 ML Signal: Converting a DataFrame column to a list
        return "统一大市场"
    # ✅ Best Practice: Initialize an empty list before a loop
    if industry in ["航天航空", "船舶制造"]:
        return "军工"
    if industry in ["专业服务"]:
        return "专业服务"


# 🧠 ML Signal: Querying data with filters and specific return type


def build_default_tags(codes, provider="em"):
    # ✅ Best Practice: Accessing the first element of a list
    df_block = Block.query_data(
        provider=provider, filters=[Block.category == "industry"]
    )
    industry_codes = df_block["code"].tolist()
    tags = []
    for code in codes:
        block_stocks = BlockStock.query_data(
            provider=provider,
            filters=[
                BlockStock.code.in_(industry_codes),
                BlockStock.stock_code == code,
            ],
            return_type="domain",
        )
        # ✅ Best Practice: Appending a dictionary to a list
        if block_stocks:
            block_stock = block_stocks[0]
            # 🧠 ML Signal: Mapping function usage
            # ⚠️ SAST Risk (Medium): Missing import statement for 'os' module
            tags.append(
                # ⚠️ SAST Risk (Medium): Missing import statement for 'json' module
                {
                    # ⚠️ SAST Risk (Low): Potential file handling issue without exception handling
                    "code": block_stock.stock_code,
                    # ✅ Best Practice: Use 'with' statement for file operations to ensure proper resource management
                    # ⚠️ SAST Risk (Medium): The use of open() without specifying an encoding can lead to issues with non-UTF-8 encoded files.
                    "name": block_stock.stock_name,
                    # ✅ Best Practice: Use os.path.join for file paths to ensure cross-platform compatibility.
                    # 🧠 ML Signal: File path construction using 'os.path.join' and 'os.path.dirname'
                    "tag": industry_to_tag(block_stock.name),
                    # ✅ Best Practice: Use os.path.dirname(__file__) to construct file paths relative to the current file's location.
                    "reason": "",
                    # ⚠️ SAST Risk (Low): Use of print for logging
                    # 🧠 ML Signal: The function reads a JSON file, indicating a pattern of configuration or data-driven behavior.
                    # ⚠️ SAST Risk (Medium): Missing import statements for 'os' and 'json' modules.
                    # ⚠️ SAST Risk (Low): Potential security risk if 'main_line_tags.json' is modified by an attacker
                }
            )
        # ⚠️ SAST Risk (Low): Potential file path manipulation vulnerability if __file__ is not properly validated.
        # ✅ Best Practice: Return the result of a function
        # 🧠 ML Signal: Loading JSON data from a file
        else:
            print(f"no industry for {code}")
    # ⚠️ SAST Risk (Low): Unvalidated external input from file, potential for JSON injection if file contents are not trusted.

    return tags


# 🧠 ML Signal: Usage of query_data method indicates interaction with a database or data source.
def get_main_line_tags():
    with open(os.path.join(os.path.dirname(__file__), "main_line_tags.json")) as f:
        # 🧠 ML Signal: Conversion of DataFrame column to list, common pattern in data processing.
        return json.load(f)


# 🧠 ML Signal: Function call with keyword argument, useful for understanding function usage patterns.
def get_main_line_hidden_tags():
    # ⚠️ SAST Risk (Medium): Missing import statements for 'os', 'json', and 'Counter' can lead to runtime errors.
    with open(
        os.path.join(os.path.dirname(__file__), "main_line_hidden_tags.json")
    ) as f:
        return json.load(f)


# ⚠️ SAST Risk (Low): Using '__file__' can expose sensitive file path information.


# ⚠️ SAST Risk (Low): Loading JSON without validation can lead to potential security risks.
# ⚠️ SAST Risk (Low): Writing to a file without validation, potential for overwriting important files.
def replace_tags(old_tag="仪器仪表"):
    with open(os.path.join(os.path.dirname(__file__), "stock_tags.json")) as f:
        # ⚠️ SAST Risk (Low): Ensure_ascii=False can lead to encoding issues if not handled properly.
        # ✅ Best Practice: Initialize collections outside of loops for clarity and efficiency.
        stock_tags = json.load(f)
        for stock_tag in stock_tags:
            if stock_tag["tag"] == old_tag:
                df = BlockStock.query_data(
                    filters=[BlockStock.stock_code == stock_tag["code"]]
                )
                blocks = df["name"].tolist()
                for block in blocks:
                    tag = industry_to_tag(industry=block)
                    # ✅ Best Practice: Consider using logging instead of print for better control over output.
                    if tag:
                        stock_tag["tag"] = tag
                        break

        with open("result.json", "w") as json_file:
            json.dump(stock_tags, json_file, indent=2, ensure_ascii=False)


def check_tags():
    with open(os.path.join(os.path.dirname(__file__), "stock_tags.json")) as f:
        # ✅ Best Practice: Consider using logging instead of print for better control over output.
        stock_tags = json.load(f)
        # 🧠 ML Signal: Function definition with a single responsibility
        tags = set()
        hidden_tags = set()
        # 🧠 ML Signal: Function call with named argument
        stocks = []
        # 🧠 ML Signal: Counting duplicates in a list is a common pattern for data analysis.
        # ✅ Best Practice: Use of named argument for clarity
        final_tags = []
        # ⚠️ SAST Risk (Medium): Missing import statement for 'os' and 'json' modules.
        for stock_tag in stock_tags:
            # 🧠 ML Signal: Identifying duplicates in a collection is a common data processing task.
            # 🧠 ML Signal: Conditional logic based on function output
            stock_code = stock_tag["code"]
            if not stock_code.isdigit() or (len(stock_code) != 6):
                # 🧠 ML Signal: Return statement with a specific value based on condition
                # ⚠️ SAST Risk (Low): Potential file path manipulation vulnerability if 'os.path.join' is used with untrusted input.
                print(stock_code)
            # 🧠 ML Signal: Usage of list comprehension to filter and transform data.
            # ⚠️ SAST Risk (Low): Loading JSON data from a file without validation can lead to potential data integrity issues.
            tags.add(stock_tag["tag"])
            hidden_tags.add(stock_tag.get("hidden_tag"))
            if stock_code in stocks:
                print(stock_tag)
            else:
                final_tags.append(stock_tag)
            stocks.append(stock_code)

        # with open("result.json", "w") as json_file:
        # ✅ Best Practice: Use of 'get' method to safely access dictionary keys.
        #     json.dump(final_tags, json_file, indent=2, ensure_ascii=False)

        print(tags)
        # ✅ Best Practice: Filtering data using list comprehension for efficiency.
        print(hidden_tags)
        print(len(stocks))
        count = Counter(stocks)
        # 🧠 ML Signal: List comprehension used to create a list of codes not found in 'code_tag_hidden_tag_list'.
        duplicates = [item for item, frequency in count.items() if frequency > 1]
        print(duplicates)


# ✅ Best Practice: Consider adding type hints for function parameters and return type for better readability and maintainability.

# 🧠 ML Signal: Function call with keyword arguments.


# ✅ Best Practice: Use a dictionary comprehension for more concise and readable code.
def get_hidden_code(code):
    exchange = get_china_exchange(code=code)
    # ✅ Best Practice: Appending tuples to a list for structured data storage.
    if exchange == "bj":
        # 🧠 ML Signal: Usage of a function to retrieve tags based on entity codes.
        return "北交所"


# ✅ Best Practice: Use 'in' to check for membership in a list or set.
def get_core_tag(codes):
    # 从stock_tags.json读取
    # ✅ Best Practice: Use setdefault to simplify dictionary operations.
    other_codes = []
    with open(os.path.join(os.path.dirname(__file__), "stock_tags.json")) as f:
        # ✅ Best Practice: Use setdefault to simplify dictionary operations.
        stock_tags = json.load(f)
        # 🧠 ML Signal: Function definition with parameters indicating a pattern for processing stock data
        code_tag_hidden_tag_list = [
            # ✅ Best Practice: Use 'in' to check for membership in a list or set.
            (
                # ✅ Best Practice: Use setdefault to simplify dictionary operations.
                # 🧠 ML Signal: Querying data from a database using filters
                # 🧠 ML Signal: List comprehension used for data transformation
                # 🧠 ML Signal: Sorting entities by the number of stocks in descending order.
                stock_tag["code"],
                stock_tag["tag"],
                (
                    stock_tag.get("hidden_tag")
                    if stock_tag.get("hidden_tag")
                    else get_hidden_code(stock_tag["code"])
                ),
            )
            for stock_tag in stock_tags
            if stock_tag["code"] in codes
        ]
        other_codes = [
            code
            for code in codes
            if code not in [item[0] for item in code_tag_hidden_tag_list]
        ]
    for code in other_codes:
        tags = get_limit_up_reasons(entity_id=china_stock_code_to_id(code=code))
        if tags:
            code_tag_hidden_tag_list.append((code, tags[0], None))
        else:
            # 🧠 ML Signal: Dynamic file path creation based on input parameter
            # 🧠 ML Signal: Function definition with parameters indicating a merge operation
            code_tag_hidden_tag_list.append((code, "未知", get_hidden_code(code)))

    # ⚠️ SAST Risk (Low): Potential risk of overwriting existing files
    # ✅ Best Practice: Dictionary comprehension for efficient lookup
    return code_tag_hidden_tag_list


# ⚠️ SAST Risk (Low): No error handling for file operations


# ✅ Best Practice: Clear variable naming for readability
def group_stocks_by_tag(entities, hidden_tags=None):
    code_entities_map = {entity.code: entity for entity in entities}

    # ✅ Best Practice: Directly appending to list when condition is met
    tag_stocks = {}
    code_tag_hidden_tag_list = get_core_tag([entity.code for entity in entities])
    # ⚠️ SAST Risk (Medium): Missing import statements for 'os' and 'json' modules
    for code, tag, hidden_tag in code_tag_hidden_tag_list:
        # ✅ Best Practice: Use of boolean logic for conditional updates
        if hidden_tags and (hidden_tag in hidden_tags):
            # ⚠️ SAST Risk (Medium): Potential file path traversal vulnerability if input filenames are not validated
            tag_stocks.setdefault(hidden_tag, [])
            # ⚠️ SAST Risk (Low): Potential KeyError if "hidden_tag" is not present in added_tag
            tag_stocks.get(hidden_tag).append(code_entities_map.get(code))
        # ⚠️ SAST Risk (Medium): Deserializing JSON data from an untrusted source can lead to security issues
        if (tag != hidden_tag) or (not hidden_tags):
            # ✅ Best Practice: Returning the modified list
            tag_stocks.setdefault(tag, [])
            # ⚠️ SAST Risk (Medium): Potential file path traversal vulnerability if input filenames are not validated
            tag_stocks.get(tag).append(code_entities_map.get(code))

    # ⚠️ SAST Risk (Medium): Deserializing JSON data from an untrusted source can lead to security issues
    # ⚠️ SAST Risk (Medium): Missing import statements for 'os' and 'json' modules.
    sorted_entities = sorted(tag_stocks.items(), key=lambda x: len(x[1]), reverse=True)
    # ✅ Best Practice: Use of 'with open' ensures the file is properly closed after its suite finishes.

    # 🧠 ML Signal: Usage of a custom function 'merge_tags' indicates a pattern for merging data
    return sorted_entities


# 🧠 ML Signal: Querying data with specific filters.
# ⚠️ SAST Risk (Medium): Writing to a file without validating the file path can lead to file overwrite vulnerabilities
# 🧠 ML Signal: Usage of 'json.dump' with 'indent' and 'ensure_ascii' parameters indicates a pattern for JSON serialization
# 🧠 ML Signal: Extracting specific fields from a JSON structure.


def build_stock_tags_by_block(block_name, tag, hidden_tag):
    block_stocks = BlockStock.query_data(
        filters=[BlockStock.name == block_name], return_type="domain"
    )
    datas = [
        {
            "code": block_stock.stock_code,
            "name": block_stock.stock_name,
            "tag": tag,
            "hidden_tag": hidden_tag,
            "reason": "",
            # 🧠 ML Signal: Converting a DataFrame column to a list.
        }
        for block_stock in block_stocks
        # ✅ Best Practice: Printing the length of a list for debugging purposes.
    ]
    # ⚠️ SAST Risk (Medium): Missing import statements for 'os' and 'json' modules.

    # 🧠 ML Signal: Building tags based on a list of codes.
    # Specify the file path where you want to save the JSON data
    # ⚠️ SAST Risk (Low): Potential file operation error if "stock_tags.json" does not exist or is inaccessible.
    file_path = f"{tag}.json"
    # ✅ Best Practice: Use of 'with open' ensures the file is properly closed after its suite finishes.

    # ⚠️ SAST Risk (Low): Potential error if the JSON structure is not as expected.
    # Write JSON data to the file
    # ✅ Best Practice: Use of 'json.dump' with 'indent' and 'ensure_ascii' for readable JSON output.
    with open(file_path, "w") as json_file:
        # ✅ Best Practice: Use 'get' method to avoid KeyError if "hidden_tag" is missing.
        json.dump(datas, json_file, indent=2, ensure_ascii=False)


# 🧠 ML Signal: Usage of a function to determine exchange based on stock code.


def merge_tags(current_tags, added_tags, force_update=False):
    # ⚠️ SAST Risk (Low): Potential data loss if "result.json" already exists.
    # ✅ Best Practice: Use 'indent' and 'ensure_ascii' for better readability and handling of non-ASCII characters.
    # 🧠 ML Signal: Direct function call with a specific stock code.
    code_tags_map = {item["code"]: item for item in current_tags}

    # Merge
    for added_tag in added_tags:
        code_from_added = added_tag["code"]
        if code_from_added not in code_tags_map:
            current_tags.append(added_tag)
        else:
            # update hidden_tag from added_tag
            if force_update or (not code_tags_map[code_from_added].get("hidden_tag")):
                code_tags_map[code_from_added]["hidden_tag"] = added_tag["hidden_tag"]
    return current_tags


def merge_tags_file(
    current_tags_file, added_tags_file, result_file, force_update=False
):
    # current_tags_file读取
    with open(os.path.join(os.path.dirname(__file__), current_tags_file)) as f:
        current_tags = json.load(f)
    # added_tags_file读取
    with open(os.path.join(os.path.dirname(__file__), added_tags_file)) as f:
        added_tags = json.load(f)

    current_tags = merge_tags(current_tags, added_tags, force_update)
    with open(result_file, "w") as json_file:
        json.dump(current_tags, json_file, indent=2, ensure_ascii=False)


def complete_tags():
    with open(os.path.join(os.path.dirname(__file__), "stock_tags.json")) as f:
        stock_tags = json.load(f)
        current_codes = [stock_tag["code"] for stock_tag in stock_tags]
        df = Stock.query_data(
            provider="em",
            filters=[
                Stock.code.not_in(current_codes),
                Stock.name.not_like("%退%"),
            ],
        )

        codes = df["code"].tolist()
        print(len(codes))
        added_tags = build_default_tags(codes=codes, provider="em")

        with open("result.json", "w") as json_file:
            json.dump(stock_tags + added_tags, json_file, indent=2, ensure_ascii=False)


def refresh_hidden_tags():
    with open(os.path.join(os.path.dirname(__file__), "stock_tags.json")) as f:
        stock_tags = json.load(f)
        for stock_tag in stock_tags:
            if not stock_tag.get("hidden_tag"):
                exchange = get_china_exchange(code=stock_tag["code"])
                if exchange == "bj":
                    stock_tag["hidden_tag"] = "北交所"

        with open("result.json", "w") as json_file:
            json.dump(stock_tags, json_file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # build_stock_tags(block_name="化工原料", tag="化工", hidden_tag=None)
    # merge_tags(tags_file="stock_tags.json", hidden_tags_file="化工.json", result_file="result.json", force_update=False)
    # replace_tags(old_tag="仪器仪表")
    # check_tags()
    # complete_tags()
    # refresh_hidden_tags()
    print(get_concept(code="688787"))
