# -*- coding: utf-8 -*-
# 🧠 ML Signal: Importing specific classes or functions indicates usage patterns and dependencies
# 🧠 ML Signal: Function definition with a specific naming pattern
from zvt.factors.zen.zen_factor import ZenFactor
# 🧠 ML Signal: Instantiation of ZenFactor with specific parameters
# ✅ Best Practice: Importing only necessary components improves code readability and maintainability


def test_zen_factor():
    z = ZenFactor(
        codes=["000338"],
        need_persist=False,
        # 🧠 ML Signal: Method call on an object with specific parameters
        provider="joinquant",
    )
    z.draw(show=True)

    z = ZenFactor(
        # 🧠 ML Signal: Re-instantiation of ZenFactor with different parameters
        # 🧠 ML Signal: Method call on an object with specific parameters
        codes=["000338", "601318"],
        need_persist=True,
        provider="joinquant",
    )
    z.draw(show=True)