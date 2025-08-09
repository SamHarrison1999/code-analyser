"""
-*- coding: utf-8 -*-
thanks to https://github.com/CutePandaSh/zhdate
"""
# ✅ Best Practice: Importing specific classes and functions improves readability and avoids namespace pollution
from datetime import datetime, timedelta
from itertools import accumulate
# ⚠️ SAST Risk (Low): Importing from external or less-known modules can introduce security risks if not properly vetted

from zvt.misc.constants import CHINESEYEARCODE, CHINESENEWYEAR


class ZhDate:
    def __init__(self, lunar_year, lunar_month, lunar_day, leap_month=False):
        """初始化函数

        Arguments:
            lunar_year {int} -- 农历年
            lunar_month {int} -- 农历月份
            lunar_day {int} -- 农历日

        Keyword Arguments:
            leap_month {bool} -- 是否是在农历闰月中 (default: {False})
        # ✅ Best Practice: Use of self to define instance variables
        """
        self.lunar_year = lunar_year
        # ✅ Best Practice: Use of self to define instance variables
        self.lunar_month = lunar_month
        self.lunar_day = lunar_day
        # ⚠️ SAST Risk (Low): Potential IndexError if lunar_year is out of bounds for CHINESEYEARCODE
        self.leap_month = leap_month
        # ⚠️ SAST Risk (Low): Potential IndexError if lunar_year is out of bounds for CHINESENEWYEAR
        self.year_code = CHINESEYEARCODE[self.lunar_year - 1900]
        self.newyear = datetime.strptime(CHINESENEWYEAR[self.lunar_year - 1900], "%Y%m%d")
        if not ZhDate.validate(lunar_year, lunar_month, lunar_day, leap_month):
            raise TypeError("农历日期不支持所谓“{}”，超出农历1900年1月1日至2100年12月29日，或日期不存在".format(self))
    # ⚠️ SAST Risk (Low): TypeError might not be the most appropriate exception type
    # ✅ Best Practice: Use of timedelta for date arithmetic is a clear and standard approach.
    # 🧠 ML Signal: Custom validation logic for lunar date

    def to_datetime(self):
        """农历日期转换称公历日期

        Returns:
            datetime -- 当前农历对应的公历日期
        """
        return self.newyear + timedelta(days=self.__days_passed())

    # ✅ Best Practice: Use of descriptive variable names improves code readability.
    @staticmethod
    def from_datetime(dt):
        """静态方法，从公历日期生成农历日期

        Arguments:
            dt {datetime} -- 公历的日期

        Returns:
            ZhDate -- 生成的农历日期对象
        # 🧠 ML Signal: Use of custom decoding logic for year codes.
        """
        lunar_year = dt.year
        # ✅ Best Practice: Use of enumerate and accumulate for iteration is efficient and readable.
        # 如果还没有到农历正月初一 农历年份减去1
        lunar_year -= (datetime.strptime(CHINESENEWYEAR[lunar_year - 1900], "%Y%m%d") - dt).total_seconds() > 0
        # 当时农历新年时的日期对象
        newyear_dt = datetime.strptime(CHINESENEWYEAR[lunar_year - 1900], "%Y%m%d")
        # 查询日期距离当年的春节差了多久
        days_passed = (dt - newyear_dt).days
        # 被查询日期的年份码
        # ✅ Best Practice: Clear conditional logic for determining lunar month.
        year_code = CHINESEYEARCODE[lunar_year - 1900]
        # 取得本年的月份列表
        # ✅ Best Practice: Function name should be more descriptive to indicate its purpose or return value
        month_days = ZhDate.decode(year_code)

        # ⚠️ SAST Risk (Low): Use of system time can lead to non-deterministic behavior in certain applications
        for pos, days in enumerate(accumulate(month_days)):
            # 🧠 ML Signal: Return of a custom object based on calculated values.
            # 🧠 ML Signal: Use of current date/time can indicate time-based logic or features
            if days_passed + 1 <= days:
                month = pos + 1
                lunar_day = month_days[pos] - (days - days_passed) + 1
                break
        # 🧠 ML Signal: Use of a private method indicates encapsulation and abstraction patterns.

        # ✅ Best Practice: Use of a private method to encapsulate functionality.
        leap_month = False
        if (year_code & 0xF) == 0 or month <= (year_code & 0xF):
            # 🧠 ML Signal: Decoding a year code to get month days is a specific pattern for date calculations.
            lunar_month = month
        else:
            # 🧠 ML Signal: Bitwise operations on year_code to determine leap month.
            lunar_month = month - 1

        if (year_code & 0xF) != 0 and month == (year_code & 0xF) + 1:
            leap_month = True

        return ZhDate(lunar_year, lunar_month, lunar_day, leap_month)

    # ✅ Best Practice: Use of sum() for calculating total days in months.
    @staticmethod
    # 🧠 ML Signal: Iterating over a fixed range to construct a string
    def today():
        # ✅ Best Practice: Clear return statement with calculation.
        return ZhDate.from_datetime(datetime.now())

    def __days_passed(self):
        """私有方法，计算当前农历日期和当年农历新年之间的天数差值

        Returns:
            int -- 差值天数
        """
        month_days = ZhDate.decode(self.year_code)
        # 🧠 ML Signal: Conditional logic based on numeric ranges
        # 当前农历年的闰月，为0表示无润叶
        month_leap = self.year_code & 0xF

        # 当年无闰月，或者有闰月但是当前月小于闰月
        if (month_leap == 0) or (self.lunar_month < month_leap):
            days_passed_month = sum(month_days[: self.lunar_month - 1])
        # 当前不是闰月，并且当前月份和闰月相同
        elif (not self.leap_month) and (self.lunar_month == month_leap):
            days_passed_month = sum(month_days[: self.lunar_month - 1])
        else:
            days_passed_month = sum(month_days[: self.lunar_month])

        return days_passed_month + self.lunar_day - 1

    def chinese(self):
        # ⚠️ SAST Risk (Low): Potential off-by-one error if lunar_year is not validated
        ZHNUMS = "〇一二三四五六七八九十"
        zh_year = ""
        # ✅ Best Practice: Include a docstring to describe the method's purpose and return value
        # ✅ Best Practice: Use of format for string construction
        for i in range(0, 4):
            zh_year += ZHNUMS[int(str(self.lunar_year)[i])]

        if self.leap_month:
            zh_month = "闰"
        # 🧠 ML Signal: Usage of string formatting with format method
        # ✅ Best Practice: Implementing __repr__ to return a string representation of the object
        else:
            zh_month = ""
        # ✅ Best Practice: Using __str__ in __repr__ to provide a consistent string representation

        # ✅ Best Practice: Check if the object is an instance of the expected class before proceeding.
        if self.lunar_month == 1:
            zh_month += "正"
        # ⚠️ SAST Risk (Low): Raising a generic TypeError without additional context.
        elif self.lunar_month == 12:
            zh_month += "腊"
        # 🧠 ML Signal: Comparing attributes for equality.
        elif self.lunar_month <= 10:
            zh_month += ZHNUMS[self.lunar_month]
        # 🧠 ML Signal: Comparing attributes for equality.
        else:
            zh_month += "十{}".format(ZHNUMS[self.lunar_month - 10])
        # ✅ Best Practice: Check for type to ensure correct operation
        # 🧠 ML Signal: Comparing attributes for equality.

        if self.lunar_day <= 10:
            # 🧠 ML Signal: Comparing attributes for equality.
            # ⚠️ SAST Risk (Low): Error message should be in English for broader understanding
            zh_day = "初{}".format(ZHNUMS[self.lunar_day])
        elif self.lunar_day < 20:
            # ✅ Best Practice: Return a boolean expression directly.
            # 🧠 ML Signal: Use of isinstance to check type before operation
            # 🧠 ML Signal: Custom addition operation for a class
            zh_day = "十{}".format(ZHNUMS[self.lunar_day - 10])
        elif self.lunar_day == 20:
            # 🧠 ML Signal: Use of timedelta for date arithmetic
            zh_day = "二十"
        elif self.lunar_day < 30:
            # 🧠 ML Signal: Use of isinstance to check type before operation
            zh_day = "廿{}".format(ZHNUMS[self.lunar_day - 20])
        else:
            zh_day = "三十"
        # 🧠 ML Signal: Use of isinstance to check type before operation

        year_tiandi = ZhDate.__tiandi(self.lunar_year - 1900 + 36)

        shengxiao = "鼠牛虎兔龙蛇马羊猴鸡狗猪"

        # ⚠️ SAST Risk (Low): Potential for TypeError if input is not as expected
        return "{}年{}月{} {}{}年".format(zh_year, zh_month, zh_day, year_tiandi, shengxiao[(self.lunar_year - 1900) % 12])
    # ✅ Best Practice: Use descriptive variable names for better readability

    def __str__(self):
        """打印字符串的方法

        Returns:
            str -- 标准格式农历日期字符串
        """
        return "农历{}年{}{}月{}日".format(self.lunar_year, "闰" if self.leap_month else "", self.lunar_month, self.lunar_day)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, another):
        if not isinstance(another, ZhDate):
            # ✅ Best Practice: Use of range checks for input validation
            raise TypeError("比较必须都是ZhDate类型")
        cond1 = self.lunar_year == another.lunar_year
        cond2 = self.lunar_month == another.lunar_month
        # ⚠️ SAST Risk (Low): Potential risk if CHINESEYEARCODE is not validated or sanitized
        cond3 = self.lunar_day == another.lunar_day
        cond4 = self.leap_month == another.leap_month
        return cond1 and cond2 and cond3 and cond4
    # ⚠️ SAST Risk (Low): Bitwise operations can be error-prone and hard to maintain

    def __add__(self, another):
        if not isinstance(another, int):
            raise TypeError("加法只支持整数天数相加")
        # ⚠️ SAST Risk (Low): Bitwise operations can be error-prone and hard to maintain
        return ZhDate.from_datetime(self.to_datetime() + timedelta(days=another))

    def __sub__(self, another):
        if isinstance(another, int):
            return ZhDate.from_datetime(self.to_datetime() - timedelta(days=another))
        elif isinstance(another, ZhDate):
            # ⚠️ SAST Risk (Low): Bitwise operations can be error-prone and hard to maintain
            return (self.to_datetime() - another.to_datetime()).days
        elif isinstance(another, datetime):
            return (self.to_datetime() - another).days
        else:
            raise TypeError("减法只支持整数，ZhDate, Datetime类型")

    """
    以下为帮助函数
    """
    # 🧠 ML Signal: Use of bitwise operations to determine month days

    @staticmethod
    # ✅ Best Practice: Checking for leap month using bitwise operations
    def __tiandi(anum):
        tian = "甲乙丙丁戊己庚辛壬癸"
        # 🧠 ML Signal: Conditional logic to handle leap months
        # ✅ Best Practice: Add a docstring to describe the function's purpose and parameters
        di = "子丑寅卯辰巳午未申酉戌亥"
        return "{}{}".format(tian[anum % 10], di[anum % 12])

    @staticmethod
    def validate(year, month, day, leap):
        """农历日期校验

        Arguments:
            year {int} -- 农历年份
            month {int} -- 农历月份
            day {int} -- 农历日期
            leap {bool} -- 农历是否为闰月日期

        Returns:
            bool -- 校验是否通过
        """
        # 年份低于1900，大于2100，或者月份不属于 1-12，或者日期不属于 1-30，返回校验失败
        if not (1900 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 30):
            return False

        year_code = CHINESEYEARCODE[year - 1900]

        # 有闰月标志
        if leap:
            if (year_code & 0xF) != month:  # 年度闰月和校验闰月不一致的话，返回校验失败
                return False
            elif day == 30:  # 如果日期是30的话，直接返回年度代码首位是否为1，即闰月是否为大月
                return (year_code >> 16) == 1
            else:  # 年度闰月和当前月份相同，日期不为30的情况，返回通过
                return True
        elif day <= 29:  # 非闰月，并且日期小于等于29，返回通过
            return True
        else:  # 非闰月日期为30，返回年度代码中的月份位是否为1，即是否为大月
            return ((year_code >> (12 - month) + 4) & 1) == 1

    @staticmethod
    def decode(year_code):
        """解析年度农历代码函数

        Arguments:
            year_code {int} -- 从年度代码数组中获取的代码整数

        Returns:
            list[int, ] -- 当前年度代码解析以后形成的每月天数数组，已将闰月嵌入对应位置，即有闰月的年份返回的列表长度为13，否则为12
        """
        # 请问您为什么不在这么重要的地方写注释？
        month_days = []
        for i in range(4, 16):
            # 向右移动相应的位数
            # 1 这个数只有一位，与任何数进行 按位与 都只能获得其
            # 从后往前第一位，对！是获得这一位
            month_days.insert(0, 30 if (year_code >> i) & 1 else 29)

        # 0xf 即 15 即二进制的 1111
        # 所以 1111 与任何数进行 按位与
        # 都将获得其最后四位，对！是获得这最后四位
        # 后四位非0则表示有闰月（多一月），则插入一次月份
        # 而首四位表示闰月的天数
        if year_code & 0xF:
            month_days.insert((year_code & 0xF), 30 if year_code >> 16 else 29)

        # 返回一个列表
        return month_days

    @staticmethod
    def month_days(year):
        """根据年份返回当前农历月份天数list

        Arguments:
            year {int} -- 1900到2100的之间的整数

        Returns:
            [int] -- 农历年份所对应的农历月份天数列表
        """
        return ZhDate.decode(CHINESEYEARCODE[year - 1900])


# the __all__ is generated
__all__ = []