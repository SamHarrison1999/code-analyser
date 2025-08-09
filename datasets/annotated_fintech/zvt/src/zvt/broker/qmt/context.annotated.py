# -*- coding: utf-8 -*-
# 🧠 ML Signal: Importing specific modules from a package indicates usage patterns
from typing import Optional

# 🧠 ML Signal: Importing specific classes from a module indicates usage patterns
from zvt import zvt_env

# ✅ Best Practice: Inheriting from 'object' is unnecessary in Python 3
from zvt.broker.qmt.qmt_account import QmtStockAccount

# ✅ Best Practice: Initialize instance variables in the constructor for clarity and maintainability.


# 🧠 ML Signal: Instantiation of a class object, which can be used to understand object creation patterns.
class QmtContext(object):
    # ✅ Best Practice: Use of default arguments to allow flexibility in function calls
    def __init__(self):
        self.qmt_account: Optional[QmtStockAccount] = None


# ⚠️ SAST Risk (Low): Potential use of unvalidated external configuration data


# ⚠️ SAST Risk (Low): Potential use of unvalidated external configuration data
qmt_context = QmtContext()


# 🧠 ML Signal: Pattern of initializing an account with specific parameters
def init_qmt_account(qmt_mini_data_path=None, qmt_account_id=None):
    # 🧠 ML Signal: Function call without arguments to rely on default behavior
    # ✅ Best Practice: Use of __all__ to define public API of the module
    if not qmt_mini_data_path:
        qmt_mini_data_path = zvt_env["qmt_mini_data_path"]
    if not qmt_account_id:
        qmt_account_id = zvt_env["qmt_account_id"]
    qmt_context.qmt_account = QmtStockAccount(
        path=qmt_mini_data_path,
        account_id=qmt_account_id,
        trader_name="zvt",
        session_id=None,
    )


init_qmt_account()


# the __all__ is generated
__all__ = ["QmtContext", "init_qmt_account"]
