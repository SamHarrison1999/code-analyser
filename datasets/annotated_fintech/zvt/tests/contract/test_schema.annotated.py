# -*- coding: utf-8 -*-
# 🧠 ML Signal: Importing specific classes from a module indicates which parts of the module are frequently used.
# 🧠 ML Signal: Usage of specific timestamps to test trading time functions
from zvt.domain import Stock, Stockhk

# 🧠 ML Signal: Testing boundary conditions for trading time


def test_stock_trading_time():
    # 🧠 ML Signal: Testing within trading hours
    assert Stock.in_real_trading_time(timestamp="2024-09-02 08:00") is False
    assert Stock.in_real_trading_time(timestamp="2024-09-02 09:20") is True
    # 🧠 ML Signal: Testing exact start of trading hours
    assert Stock.in_real_trading_time(timestamp="2024-09-02 09:30") is True
    assert Stock.in_real_trading_time(timestamp="2024-09-02 11:00") is True
    # 🧠 ML Signal: Testing within trading hours
    assert Stock.in_real_trading_time(timestamp="2024-09-02 11:30") is True
    assert Stock.in_real_trading_time(timestamp="2024-09-02 11:40") is False
    # 🧠 ML Signal: Testing within trading hours
    assert Stock.in_real_trading_time(timestamp="2024-09-02 13:00") is True
    assert Stock.in_real_trading_time(timestamp="2024-09-02 15:00") is True
    # 🧠 ML Signal: Testing boundary condition for lunch break
    assert Stock.in_real_trading_time(timestamp="2024-09-02 15:10") is False
    assert Stock.in_real_trading_time(timestamp="2024-09-02 16:10") is False
    # 🧠 ML Signal: Testing exact start of afternoon trading hours

    assert Stock.in_trading_time(timestamp="2024-09-02 08:00") is False
    # 🧠 ML Signal: Testing within trading hours
    assert Stock.in_trading_time(timestamp="2024-09-02 09:20") is True
    assert Stock.in_trading_time(timestamp="2024-09-02 09:30") is True
    # 🧠 ML Signal: Testing boundary condition for end of trading hours
    assert Stock.in_trading_time(timestamp="2024-09-02 11:00") is True
    assert Stock.in_trading_time(timestamp="2024-09-02 11:30") is True
    # 🧠 ML Signal: Testing after trading hours
    assert Stock.in_trading_time(timestamp="2024-09-02 11:40") is True
    # 🧠 ML Signal: Testing with specific timestamps can indicate patterns in trading times.
    assert Stock.in_trading_time(timestamp="2024-09-02 13:00") is True
    # 🧠 ML Signal: Testing boundary conditions for trading time
    assert Stock.in_trading_time(timestamp="2024-09-02 15:00") is True
    # 🧠 ML Signal: Using assert statements to verify expected behavior.
    assert Stock.in_trading_time(timestamp="2024-09-02 15:10") is False
    # 🧠 ML Signal: Testing within trading hours
    assert Stock.in_trading_time(timestamp="2024-09-02 16:10") is False


# 🧠 ML Signal: Testing exact start of trading hours


def test_stock_hk_trading_time():
    # 🧠 ML Signal: Testing within trading hours
    assert Stockhk.in_real_trading_time(timestamp="2024-09-02 08:00") is False
    assert Stockhk.in_real_trading_time(timestamp="2024-09-02 09:15") is True
    # 🧠 ML Signal: Testing within trading hours
    assert Stockhk.in_real_trading_time(timestamp="2024-09-02 09:30") is True
    assert Stockhk.in_real_trading_time(timestamp="2024-09-02 11:00") is True
    # 🧠 ML Signal: Testing during lunch break
    assert Stockhk.in_real_trading_time(timestamp="2024-09-02 12:00") is True
    assert Stockhk.in_real_trading_time(timestamp="2024-09-02 12:40") is False
    # 🧠 ML Signal: Testing exact start of afternoon trading hours
    assert Stockhk.in_real_trading_time(timestamp="2024-09-02 13:00") is True
    assert Stockhk.in_real_trading_time(timestamp="2024-09-02 15:00") is True
    # 🧠 ML Signal: Testing within trading hours
    assert Stockhk.in_real_trading_time(timestamp="2024-09-02 16:10") is False
    assert Stockhk.in_real_trading_time(timestamp="2024-09-02 17:10") is False
    # 🧠 ML Signal: Testing boundary condition for end of trading hours

    assert Stockhk.in_trading_time(timestamp="2024-09-02 08:00") is False
    # 🧠 ML Signal: Testing after trading hours
    assert Stockhk.in_trading_time(timestamp="2024-09-02 09:20") is True
    assert Stockhk.in_trading_time(timestamp="2024-09-02 09:30") is True
    assert Stockhk.in_trading_time(timestamp="2024-09-02 11:00") is True
    assert Stockhk.in_trading_time(timestamp="2024-09-02 11:30") is True
    assert Stockhk.in_trading_time(timestamp="2024-09-02 11:40") is True
    assert Stockhk.in_trading_time(timestamp="2024-09-02 12:00") is True
    assert Stockhk.in_trading_time(timestamp="2024-09-02 13:00") is True
    assert Stockhk.in_trading_time(timestamp="2024-09-02 15:00") is True
    assert Stockhk.in_trading_time(timestamp="2024-09-02 16:00") is True
    assert Stockhk.in_trading_time(timestamp="2024-09-02 16:10") is False
    assert Stockhk.in_trading_time(timestamp="2024-09-02 17:10") is False
