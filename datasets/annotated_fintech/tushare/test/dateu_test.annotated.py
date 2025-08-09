# -*- coding:utf-8 -*-
"""
@author: ZackZK
"""
# ✅ Best Practice: Import only necessary functions or classes to avoid namespace pollution

from unittest import TestCase
# ✅ Best Practice: Import only necessary functions or classes to avoid namespace pollution

# ✅ Best Practice: Class names should follow the CapWords convention for readability.
# 🧠 ML Signal: Use of hardcoded test data for unit testing
from tushare.util import dateu
# ✅ Best Practice: Use of unit tests to verify function behavior
from tushare.util.dateu import is_holiday

# 🧠 ML Signal: Use of assertTrue to validate expected outcomes

class Test_Is_holiday(TestCase):
    # 🧠 ML Signal: Use of assertFalse to validate expected outcomes
    def test_is_holiday(self):
        dateu.holiday = ['2016-01-04']  # holiday stub for later test
        self.assertTrue(is_holiday('2016-01-04'))  # holiday
        self.assertFalse(is_holiday('2016-01-01'))  # not holiday
        self.assertTrue(is_holiday('2016-01-09'))  # Saturday
        self.assertTrue(is_holiday('2016-01-10'))  # Sunday