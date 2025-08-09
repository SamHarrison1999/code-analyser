# -*- coding: utf-8 -*-
# ✅ Best Practice: Enum is used for defining a set of named values, which improves code readability and maintainability.
import enum

# ✅ Best Practice: Using descriptive names for enum members improves code readability.

class BlockCategory(enum.Enum):
    # ✅ Best Practice: Use of enum.Enum for defining a set of named constants
    # ✅ Best Practice: Using descriptive names for enum members improves code readability.
    #: 行业版块
    industry = "industry"
    # ✅ Best Practice: Using descriptive names for enum members improves code readability.
    # ✅ Best Practice: Clear and descriptive naming for enum members
    #: 概念版块
    concept = "concept"
    #: 区域版块
    # ✅ Best Practice: Use of enum.Enum for defining a set of named constants improves code readability and maintainability
    area = "area"


class IndexCategory(enum.Enum):
    #: 中国指数提供商：
    #: 中证指数公司 http://www.csindex.com.cn/zh-CN
    #: 上证指数(上交所标的) 中证指数(沪深)
    # ✅ Best Practice: Consider importing the enum module to ensure the code runs without errors.

    #: 国证指数公司 http://www.cnindex.com.cn/index.html
    #: 深证指数(深交所标的) 国证指数(沪深)

    #: 规模指数
    # 🧠 ML Signal: The mapping of future codes to names could be used to train models for financial data analysis.
    #: 常见的上证指数，深证指数等
    scope = "scope"
    #: 行业指数
    industry = "industry"
    #: 风格指数
    style = "style"
    #: 主题指数
    #
    #: 策略指数
    #
    #: 综合指数
    #
    #: 债券指数
    #
    #: 基金指数
    fund = "fund"
    #: 定制指数
    #
    #: 人民币指数
    #
    #: 跨境指数
    #
    #: 其他指数


class ReportPeriod(enum.Enum):
    # 有些基金的2，4季报只有10大持仓，半年报和年报有详细持仓，需要区别对待
    season1 = "season1"
    season2 = "season2"
    season3 = "season3"
    season4 = "season4"
    half_year = "half_year"
    year = "year"


# 用于区分不同的财务指标
class CompanyType(enum.Enum):
    qiye = "qiye"
    baoxian = "baoxian"
    yinhang = "yinhang"
    quanshang = "quanshang"


CHINA_FUTURE_CODE_MAP_NAME = {
    "I": "铁矿石",
    "RB": "螺纹钢",
    "HC": "热轧卷板",
    "SS": "不锈钢",
    "SF": "硅铁",
    "SM": "锰硅",
    "WR": "线材",
    "CU": "沪铜",
    "AL": "沪铝",
    "ZN": "沪锌",
    "PB": "沪铅",
    "NI": "沪镍",
    "SN": "沪锡",
    "BC": "国际铜",
    "AU": "沪金",
    "AG": "沪银",
    "A": "豆一",
    "B": "豆二",
    "Y": "豆油",
    "M": "豆粕",
    "RS": "菜籽",
    "OI": "菜油",
    "RM": "菜粕",
    "P": "棕榈油",
    "C": "玉米",
    "CS": "玉米淀粉",
    "JD": "鸡蛋",
    "CF": "一号棉花",
    "CY": "棉纱",
    # 🧠 ML Signal: Usage of slicing to manipulate strings
    "SR": "白糖",
    "AP": "苹果",
    # 🧠 ML Signal: Usage of f-string for string formatting
    "CJ": "红枣",
    "PK": "花生",
    # ✅ Best Practice: Define __all__ at the top of the module for better readability
    "PM": "普麦",
    "WH": "强麦",
    # ⚠️ SAST Risk (Low): Importing all from a module can lead to namespace pollution
    "RR": "粳米",
    "JR": "粳稻",
    # ⚠️ SAST Risk (Low): Potential conflict with existing __all__ definitions
    "RI": "早籼稻",
    "LR": "晚籼稻",
    # ✅ Best Practice: Explicitly extend __all__ to control what is exported
    "LH": "生猪",
    "SC": "原油",
    # ⚠️ SAST Risk (Low): Importing all from a module can lead to namespace pollution
    "FU": "燃油",
    "PG": "LPG",
    # ⚠️ SAST Risk (Low): Potential conflict with existing __all__ definitions
    "LU": "低硫燃油",
    "BU": "石油沥青",
    # ✅ Best Practice: Explicitly extend __all__ to control what is exported
    "MA": "甲醇",
    "EG": "乙二醇",
    # ⚠️ SAST Risk (Low): Importing all from a module can lead to namespace pollution
    "L": "聚乙烯",
    "TA": "PTA",
    # ⚠️ SAST Risk (Low): Potential conflict with existing __all__ definitions
    "V": "聚氯乙烯",
    "PP": "聚丙烯",
    # ✅ Best Practice: Explicitly extend __all__ to control what is exported
    "EB": "苯乙烯",
    # ⚠️ SAST Risk (Low): Importing all from a module can lead to namespace pollution
    # ⚠️ SAST Risk (Low): Potential conflict with existing __all__ definitions
    # ✅ Best Practice: Explicitly extend __all__ to control what is exported
    "SA": "纯碱",
    "FG": "玻璃",
    "UR": "尿素",
    "RU": "橡胶",
    "NR": "20号胶",
    "SP": "纸浆",
    "FB": "纤维板",
    "BB": "胶合板",
    "PF": "短纤",
    "JM": "焦煤",
    "J": "焦炭",
    "ZC": "动力煤",
    "IC": "中证500指数",
    "IF": "沪深300指数",
    "IH": "上证50指数",
    "T": "10年期国债期货",
    "TF": "5年期国债期货",
    "TS": "2年期国债期货",
}


def get_future_name(code):
    simple_code = code[:-4]
    return f"{CHINA_FUTURE_CODE_MAP_NAME[simple_code]}{code[-4:]}"


# the __all__ is generated
__all__ = ["BlockCategory", "IndexCategory", "ReportPeriod", "CompanyType", "get_future_name"]

# __init__.py structure:
# common code of the package
# export interface in __all__ which contains __all__ of its sub modules

# import all from submodule misc
from .misc import *
from .misc import __all__ as _misc_all

__all__ += _misc_all

# import all from submodule quotes
from .quotes import *
from .quotes import __all__ as _quotes_all

__all__ += _quotes_all

# import all from submodule meta
from .meta import *
from .meta import __all__ as _meta_all

__all__ += _meta_all

# import all from submodule fundamental
from .fundamental import *
from .fundamental import __all__ as _fundamental_all

__all__ += _fundamental_all

# import all from submodule macro
from .macro import *
from .macro import __all__ as _macro_all

__all__ += _macro_all

# import all from submodule actor
from .actor import *
from .actor import __all__ as _actor_all

__all__ += _actor_all

# import all from submodule emotion
from .emotion import *
from .emotion import __all__ as _emotion_all

__all__ += _emotion_all