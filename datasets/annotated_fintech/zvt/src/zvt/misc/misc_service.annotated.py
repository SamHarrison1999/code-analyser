# -*- coding: utf-8 -*-
# ✅ Best Practice: Grouping imports from the same module together improves readability.
from zvt.misc.zhdate import ZhDate
from zvt.utils.time_utils import to_pd_timestamp, current_date, count_interval
# ✅ Best Practice: Use of default parameter values for flexibility


def holiday_distance(timestamp=None, consider_future_days=15):
    if not timestamp:
        the_date = current_date()
    # 🧠 ML Signal: Extracting month from date for conditional logic
    else:
        the_date = to_pd_timestamp(timestamp)
    # 🧠 ML Signal: Collecting information messages in a list

    # 业绩预告
    month = the_date.month
    # 🧠 ML Signal: Appending specific messages based on conditions

    infos = [f"今天是{the_date.date()}"]
    if month == 12:
        # ⚠️ SAST Risk (Low): Potential timezone issues with date calculations
        infos.append("业绩预告期，注意排雷")

        # 元旦
        new_year = to_pd_timestamp(f"{the_date.year + 1}-01-01")
        distance = count_interval(the_date, new_year)
        # 🧠 ML Signal: Use of lunar calendar for date calculations
        if 0 < distance < consider_future_days:
            infos.append(f"距离元旦还有{distance}天")
    if month in (1, 2):
        # 春节
        zh_date = ZhDate(lunar_year=the_date.year, lunar_month=1, lunar_day=1)
        spring_date = zh_date.newyear
        distance = count_interval(the_date, spring_date)
        if 0 < distance < consider_future_days:
            infos.append(f"距离春节还有{distance}天")

        # 两会
        # 三月初
        lianghui = to_pd_timestamp(f"{the_date.year}-03-01")
        distance = count_interval(the_date, lianghui)
        if 0 < distance < consider_future_days:
            infos.append(f"距离两会还有{distance}天")

    # 年报发布
    if month in (3, 4):
        infos.append("年报发布期，注意排雷")

    # ✅ Best Practice: Use of dictionary to return multiple values
    # 五一
    if month == 4:
        wuyi = to_pd_timestamp(f"{the_date.year}-05-01")
        # 🧠 ML Signal: Joining list of messages into a single string
        # 🧠 ML Signal: Common pattern for script entry point
        # ✅ Best Practice: Use of __all__ to define public API of the module
        distance = count_interval(the_date, wuyi)
        if 0 < distance < consider_future_days:
            infos.append(f"距离五一还有{distance}天")

    # 业绩发布
    if month in (7, 8):
        infos.append("半年报发布期，注意排雷")

    if month == 9:
        # 国庆
        shiyi = to_pd_timestamp(f"{the_date.year}-10-01")
        distance = count_interval(the_date, shiyi)
        if 0 < distance < consider_future_days:
            infos.append(f"距离国庆还有{distance}天")

    msg = "，".join(infos)
    return msg


def get_time_message():
    return {"timestamp": current_date(), "message": holiday_distance()}


if __name__ == "__main__":
    print(get_time_message())

# the __all__ is generated
__all__ = ["holiday_distance", "get_time_message"]