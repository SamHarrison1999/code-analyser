# -*- coding: utf-8 -*-
from sqlalchemy import Column, String, DateTime, Float, Integer

# 🧠 ML Signal: Usage of custom mixin class for ORM models
from sqlalchemy.orm import declarative_base

# 🧠 ML Signal: Custom schema registration pattern
from zvt.contract import Mixin

# ✅ Best Practice: Use of @classmethod to define a method that operates on the class itself rather than an instance
# 🧠 ML Signal: Usage of declarative base for ORM models
from zvt.contract.register import register_schema

# ✅ Best Practice: Consider using a class method decorator for clarity and convention.
FinanceBase = declarative_base()


class BalanceSheet(FinanceBase, Mixin):
    @classmethod
    def important_cols(cls):
        return [
            "total_assets",
            "total_liabilities",
            "equity",
            "cash_and_cash_equivalents",
            # 🧠 ML Signal: The use of a class variable for table name can be a pattern for ORM models.
            "accounts_receivable",
            "inventories",
            # 🧠 ML Signal: Column definitions are typical in ORM models, useful for schema inference.
            "goodwill",
        ]

    __tablename__ = "balance_sheet"

    provider = Column(String(length=32))
    code = Column(String(length=32))

    report_period = Column(String(length=32))
    report_date = Column(DateTime)

    #: 流动资产
    #
    #: 货币资金
    cash_and_cash_equivalents = Column(Float)
    #: 应收票据
    note_receivable = Column(Float)
    #: 应收账款
    accounts_receivable = Column(Float)
    #: 预付款项
    advances_to_suppliers = Column(Float)
    #: 其他应收款
    other_receivables = Column(Float)
    #: 存货
    inventories = Column(Float)
    #: 一年内到期的非流动资产
    current_portion_of_non_current_assets = Column(Float)
    #: 其他流动资产
    other_current_assets = Column(Float)
    #: 流动资产合计
    total_current_assets = Column(Float)
    #: 非流动资产
    #
    #: 可供出售金融资产
    fi_assets_saleable = Column(Float)
    #: 长期应收款
    long_term_receivables = Column(Float)
    #: 长期股权投资
    long_term_equity_investment = Column(Float)
    #: 投资性房地产
    real_estate_investment = Column(Float)
    #: 固定资产
    fixed_assets = Column(Float)
    #: 在建工程
    construction_in_process = Column(Float)
    #: 无形资产
    intangible_assets = Column(Float)
    #: 商誉
    goodwill = Column(Float)
    #: 长期待摊费用
    long_term_prepaid_expenses = Column(Float)
    #: 递延所得税资产
    deferred_tax_assets = Column(Float)
    #: 其他非流动资产
    other_non_current_assets = Column(Float)
    #: 非流动资产合计
    total_non_current_assets = Column(Float)
    #: 资产总计
    total_assets = Column(Float)
    #: 流动负债
    #
    #: 短期借款
    short_term_borrowing = Column(Float)
    #: 吸收存款及同业存放
    accept_money_deposits = Column(Float)
    #: 应付账款
    accounts_payable = Column(Float)
    #: 预收款项
    advances_from_customers = Column(Float)
    #: 应付职工薪酬
    employee_benefits_payable = Column(Float)
    #: 应交税费
    taxes_payable = Column(Float)
    #: 应付利息
    interest_payable = Column(Float)
    #: 其他应付款
    other_payable = Column(Float)
    #: 一年内到期的非流动负债
    current_portion_of_non_current_liabilities = Column(Float)
    #: 其他流动负债
    other_current_liabilities = Column(Float)
    #: 流动负债合计
    total_current_liabilities = Column(Float)
    #: 非流动负债
    #
    #: 长期借款
    long_term_borrowing = Column(Float)
    #: 长期应付款
    long_term_payable = Column(Float)
    #: 递延收益
    deferred_revenue = Column(Float)
    #: 递延所得税负债
    deferred_tax_liabilities = Column(Float)
    #: 其他非流动负债
    other_non_current_liabilities = Column(Float)
    #: 非流动负债合计
    total_non_current_liabilities = Column(Float)
    #: 负债合计
    total_liabilities = Column(Float)
    #: 所有者权益(或股东权益)
    #
    #: 实收资本（或股本）
    capital = Column(Float)
    # ✅ Best Practice: Use of classmethod decorator to define a method that operates on the class rather than instance
    #: 资本公积
    capital_reserve = Column(Float)
    # ✅ Best Practice: Consider using a class method decorator for class methods
    #: 专项储备
    # 🧠 ML Signal: Method for creating an object from a dictionary, indicating a common pattern for data deserialization
    # ✅ Best Practice: Use of class method to create an instance from a dictionary
    # 🧠 ML Signal: Instantiation of class using cls() within a class method
    special_reserve = Column(Float)
    #: 盈余公积
    surplus_reserve = Column(Float)
    #: 未分配利润
    undistributed_profits = Column(Float)
    # 🧠 ML Signal: Iterating over dictionary items, common pattern for processing key-value pairs
    #: 归属于母公司股东权益合计
    equity = Column(Float)
    # ⚠️ SAST Risk (Low): Use of setattr can lead to security risks if keys are not validated
    #: 少数股东权益
    # 🧠 ML Signal: Dynamic attribute setting using setattr, indicating flexible object construction
    equity_as_minority_interest = Column(Float)
    #: 股东权益合计
    # 🧠 ML Signal: Use of a class variable to define a database table name
    # 🧠 ML Signal: Returning an instance of the class, common pattern in factory methods
    total_equity = Column(Float)
    #: 负债和股东权益合计
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    total_liabilities_and_equity = Column(Float)
    # 🧠 ML Signal: Method for converting an object to a dictionary, indicating a common pattern for data serialization

    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    #: 银行相关
    # 🧠 ML Signal: Initialization of an empty dictionary for data collection
    #: 资产
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    #: 现金及存放中央银行款项
    # 🧠 ML Signal: Accessing object's __dict__ to iterate over its attributes
    fi_cash_and_deposit_in_central_bank = Column(Float)
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    #: 存放同业款项
    # 🧠 ML Signal: Dynamic attribute access using getattr, indicating flexible data retrieval
    fi_deposit_in_other_fi = Column(Float)
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    #: 贵金属
    # 🧠 ML Signal: Returning a dictionary representation of the object, common pattern in serialization methods
    fi_expensive_metals = Column(Float)
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    #: 拆出资金
    fi_lending_to_other_fi = Column(Float)
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    #: 以公允价值计量且其变动计入当期损益的金融资产
    fi_financial_assets_effect_current_income = Column(Float)
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    #: 衍生金融资产
    fi_financial_derivative_asset = Column(Float)
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    #: 买入返售金融资产
    fi_buying_sell_back_fi__asset = Column(Float)
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    #: 应收账款
    #
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    #: 应收利息
    fi_interest_receivable = Column(Float)
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    #: 发放贷款及垫款
    fi_disbursing_loans_and_advances = Column(Float)
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    #: 可供出售金融资产
    #
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    #: 持有至到期投资
    fi_held_to_maturity_investment = Column(Float)
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    #: 应收款项类投资
    fi_account_receivable_investment = Column(Float)
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    #: 投资性房地产
    #
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    #: 固定资产
    #
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    #: 无形资产
    #
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    #: 商誉
    #
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    #: 递延所得税资产
    #
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    #: 其他资产
    fi_other_asset = Column(Float)
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    #: 资产总计
    #
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema

    #: 负债
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    #
    #: 向中央银行借款
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    fi_borrowings_from_central_bank = Column(Float)
    #: 同业和其他金融机构存放款项
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    fi_deposit_from_other_fi = Column(Float)
    #: 拆入资金
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    fi_borrowings_from_fi = Column(Float)
    #: 以公允价值计量且其变动计入当期损益的金融负债
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    fi_financial_liability_effect_current_income = Column(Float)
    #: 衍生金融负债
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    fi_financial_derivative_liability = Column(Float)
    #: 卖出回购金融资产款
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    fi_sell_buy_back_fi_asset = Column(Float)
    #: 吸收存款
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    fi_savings_absorption = Column(Float)
    #: 存款证及应付票据
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    fi_notes_payable = Column(Float)
    #: 应付职工薪酬
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    #
    #: 应交税费
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    #
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    # ✅ Best Practice: Use of classmethod to define a method that operates on the class itself rather than instances
    #: 应付利息
    #
    # 🧠 ML Signal: Method returning a list of important column names
    #: 预计负债
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    # 🧠 ML Signal: Method for creating an object from a dictionary, indicating a common pattern for data deserialization
    fi_estimated_liabilities = Column(Float)
    # ✅ Best Practice: Use of class method for alternative constructor
    #: 应付债券
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    # 🧠 ML Signal: Table name for ORM mapping
    fi_bond_payable = Column(Float)
    #: 其他负债
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    # 🧠 ML Signal: Column definition for ORM mapping
    fi_other_liability = Column(Float)
    #: 负债合计
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    # 🧠 ML Signal: Column definition for ORM mapping
    #

    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    # 🧠 ML Signal: Column definition for ORM mapping
    #: 所有者权益(或股东权益)
    # ✅ Best Practice: Constructor clearly defines required parameters
    #: 股本
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    # 🧠 ML Signal: Column definition for ORM mapping
    fi_capital = Column(Float)
    #: 其他权益工具
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    # 🧠 ML Signal: Column definition for ORM mapping
    fi_other_equity_instruments = Column(Float)
    #: 其中:优先股
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    # 🧠 ML Signal: Column definition for ORM mapping
    fi_preferred_stock = Column(Float)
    # 🧠 ML Signal: Method calculating a total value, indicating a pattern of aggregation
    #: 资本公积
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    # 🧠 ML Signal: Column definition for ORM mapping
    #
    #: 盈余公积
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    # 🧠 ML Signal: Column definition for ORM mapping
    #
    #: 一般风险准备
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    # 🧠 ML Signal: Column definition for ORM mapping
    fi_generic_risk_reserve = Column(Float)
    # ✅ Best Practice: Implementation of __repr__ for better debugging and logging
    #: 未分配利润
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    # 🧠 ML Signal: Column definition for ORM mapping
    #
    #: 归属于母公司股东权益合计
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    # 🧠 ML Signal: Column definition for ORM mapping
    #
    #: 股东权益合计
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    # 🧠 ML Signal: Column definition for ORM mapping
    #
    #: 负债及股东权益总计
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    # 🧠 ML Signal: Column definition for ORM mapping

    #: 券商相关
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    # 🧠 ML Signal: Column definition for ORM mapping
    #: 资产
    #
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    # 🧠 ML Signal: Column definition for ORM mapping
    #: 货币资金
    #
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    # 🧠 ML Signal: Column definition for ORM mapping
    #: 其中: 客户资金存款
    fi_client_fund = Column(Float)
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    # 🧠 ML Signal: Column definition for ORM mapping
    #: 结算备付金
    fi_deposit_reservation_for_balance = Column(Float)
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    # 🧠 ML Signal: Column definition for ORM mapping
    #: 其中: 客户备付金
    fi_client_deposit_reservation_for_balance = Column(Float)
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    # 🧠 ML Signal: Column definition for ORM mapping
    #: 融出资金
    fi_margin_out_fund = Column(Float)
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    # 🧠 ML Signal: Column definition for ORM mapping
    #: 以公允价值计量且其变动计入当期损益的金融资产
    #
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    # 🧠 ML Signal: Column definition for ORM mapping
    #: 衍生金融资产
    #
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    # 🧠 ML Signal: Column definition for ORM mapping
    #: 买入返售金融资产
    #
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    # 🧠 ML Signal: Column definition for ORM mapping
    #: 应收利息
    #
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    # 🧠 ML Signal: Column definition for ORM mapping
    #: 应收款项
    fi_receivables = Column(Float)
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    # 🧠 ML Signal: Column definition for ORM mapping
    #: 存出保证金
    fi_deposit_for_recognizance = Column(Float)
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    # 🧠 ML Signal: Column definition for ORM mapping
    #: 可供出售金融资产
    #
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    # 🧠 ML Signal: Column definition for ORM mapping
    #: 持有至到期投资
    #
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    # 🧠 ML Signal: Column definition for ORM mapping
    #: 长期股权投资
    #
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    # 🧠 ML Signal: Column definition for ORM mapping
    #: 固定资产
    #
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    # 🧠 ML Signal: Column definition for ORM mapping
    #: 在建工程
    #
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    # 🧠 ML Signal: Column definition for ORM mapping
    #: 无形资产
    #
    # 🧠 ML Signal: Use of SQLAlchemy Column to define database schema
    # 🧠 ML Signal: Column definition for ORM mapping
    #: 商誉
    #
    # 🧠 ML Signal: Column definition for ORM mapping
    #: 递延所得税资产
    #
    # 🧠 ML Signal: Column definition for ORM mapping
    #: 其他资产
    #
    # 🧠 ML Signal: Column definition for ORM mapping
    #: 资产总计
    #
    # 🧠 ML Signal: Column definition for ORM mapping
    #: 负债
    #
    # 🧠 ML Signal: Column definition for ORM mapping
    #: 短期借款
    #
    # 🧠 ML Signal: Column definition for ORM mapping
    #: 拆入资金
    #
    # 🧠 ML Signal: Column definition for ORM mapping
    #: 以公允价值计量且其变动计入当期损益的金融负债
    #
    # 🧠 ML Signal: Column definition for ORM mapping
    # ✅ Best Practice: Use of @classmethod for methods that operate on the class itself rather than instances
    #: 衍生金融负债
    #
    # 🧠 ML Signal: Column definition for ORM mapping
    # ✅ Best Practice: Consider using a class method decorator for clarity and convention.
    #: 卖出回购金融资产款
    # 🧠 ML Signal: Column definition for ORM mapping
    #
    #: 代理买卖证券款
    fi_receiving_as_agent = Column(Float)
    #: 应付账款
    #
    #: 应付职工薪酬
    #
    # 🧠 ML Signal: Column definition for ORM mapping
    #: 应交税费
    #
    # 🧠 ML Signal: Column definition for ORM mapping
    #: 应付利息
    #
    # 🧠 ML Signal: Column definition for ORM mapping
    #: 应付短期融资款
    # 🧠 ML Signal: Use of a specific table name in a database schema.
    fi_short_financing_payable = Column(Float)
    # 🧠 ML Signal: Column definition for ORM mapping
    #: 预计负债
    # 🧠 ML Signal: Use of SQLAlchemy Column definitions for ORM mapping.
    #
    # 🧠 ML Signal: Column definition for ORM mapping
    #: 应付债券
    # 🧠 ML Signal: Use of SQLAlchemy Column definitions for ORM mapping.
    #
    # 🧠 ML Signal: Column definition for ORM mapping
    #: 递延所得税负债
    # 🧠 ML Signal: Use of SQLAlchemy Column definitions for ORM mapping.
    #
    # 🧠 ML Signal: Column definition for ORM mapping
    #: 其他负债
    # 🧠 ML Signal: Use of SQLAlchemy Column definitions for ORM mapping.
    #
    # 🧠 ML Signal: Column definition for ORM mapping
    #: 负债合计
    # 🧠 ML Signal: Use of SQLAlchemy Column definitions for ORM mapping.
    #
    # 🧠 ML Signal: Column definition for ORM mapping
    #: 所有者权益(或股东权益)
    # 🧠 ML Signal: Use of SQLAlchemy Column definitions for ORM mapping.
    #
    # 🧠 ML Signal: Column definition for ORM mapping
    #: 股本
    # 🧠 ML Signal: Use of SQLAlchemy Column definitions for ORM mapping.
    #
    # 🧠 ML Signal: Column definition for ORM mapping
    #: 资本公积
    # 🧠 ML Signal: Use of SQLAlchemy Column definitions for ORM mapping.
    #
    # 🧠 ML Signal: Column definition for ORM mapping
    #: 其他权益工具
    # 🧠 ML Signal: Use of SQLAlchemy Column definitions for ORM mapping.
    #
    # 🧠 ML Signal: Column definition for ORM mapping
    #: 盈余公积
    # 🧠 ML Signal: Use of SQLAlchemy Column definitions for ORM mapping.
    #
    # 🧠 ML Signal: Column definition for ORM mapping
    #: 一般风险准备
    # 🧠 ML Signal: Use of SQLAlchemy Column definitions for ORM mapping.
    #
    # 🧠 ML Signal: Column definition for ORM mapping
    #: 交易风险准备
    # 🧠 ML Signal: Use of SQLAlchemy Column definitions for ORM mapping.
    fi_trade_risk_reserve = Column(Float)
    # 🧠 ML Signal: Column definition for ORM mapping
    #: 未分配利润
    # 🧠 ML Signal: Use of SQLAlchemy Column definitions for ORM mapping.
    #
    # 🧠 ML Signal: Column definition for ORM mapping
    #: 归属于母公司股东权益合计
    # 🧠 ML Signal: Use of SQLAlchemy Column definitions for ORM mapping.
    #
    # 🧠 ML Signal: Column definition for ORM mapping
    #: 少数股东权益
    # 🧠 ML Signal: Use of SQLAlchemy Column definitions for ORM mapping.
    #
    # 🧠 ML Signal: Column definition for ORM mapping
    #: 股东权益合计
    # 🧠 ML Signal: Use of SQLAlchemy Column definitions for ORM mapping.
    #
    # 🧠 ML Signal: Column definition for ORM mapping
    #: 负债和股东权益总计
    # 🧠 ML Signal: Use of SQLAlchemy Column definitions for ORM mapping.

    # 🧠 ML Signal: Column definition for ORM mapping
    #: 保险相关
    # 🧠 ML Signal: Use of SQLAlchemy Column definitions for ORM mapping.

    # 🧠 ML Signal: Column definition for ORM mapping
    #: 资产
    # 🧠 ML Signal: Use of SQLAlchemy Column definitions for ORM mapping.
    #: 应收保费
    # 🧠 ML Signal: Column definition for ORM mapping
    fi_premiums_receivable = Column(Float)
    # 🧠 ML Signal: Use of SQLAlchemy Column definitions for ORM mapping.
    #: 应收分保账款
    # 🧠 ML Signal: Column definition for ORM mapping
    fi_reinsurance_premium_receivable = Column(Float)
    # 🧠 ML Signal: Use of SQLAlchemy Column definitions for ORM mapping.
    #: 应收分保合同准备金
    # 🧠 ML Signal: Column definition for ORM mapping
    fi_reinsurance_contract_reserve = Column(Float)
    # 🧠 ML Signal: Use of SQLAlchemy Column definitions for ORM mapping.
    #: 保户质押贷款
    # 🧠 ML Signal: Column definition for ORM mapping
    fi_policy_pledge_loans = Column(Float)
    # 🧠 ML Signal: Use of SQLAlchemy Column definitions for ORM mapping.
    #: 发放贷款及垫款
    # 🧠 ML Signal: Column definition for ORM mapping
    #: 定期存款
    # 🧠 ML Signal: Use of SQLAlchemy Column definitions for ORM mapping.
    fi_time_deposit = Column(Float)
    # 🧠 ML Signal: Column definition for ORM mapping
    #: 可供出售金融资产
    # 🧠 ML Signal: Use of SQLAlchemy Column definitions for ORM mapping.
    #
    # 🧠 ML Signal: Column definition for ORM mapping
    #: 持有至到期投资
    # 🧠 ML Signal: Use of SQLAlchemy Column definitions for ORM mapping.
    #
    # 🧠 ML Signal: Column definition for ORM mapping
    #: 应收款项类投资
    # 🧠 ML Signal: Use of SQLAlchemy Column definitions for ORM mapping.
    #
    # 🧠 ML Signal: Column definition for ORM mapping
    #: 应收账款
    # 🧠 ML Signal: Use of SQLAlchemy Column definitions for ORM mapping.
    #
    # 🧠 ML Signal: Column definition for ORM mapping
    #: 长期股权投资
    # 🧠 ML Signal: Use of SQLAlchemy Column definitions for ORM mapping.
    #
    # 🧠 ML Signal: Column definition for ORM mapping
    #: 存出资本保证金
    # 🧠 ML Signal: Use of SQLAlchemy Column definitions for ORM mapping.
    fi_deposit_for_capital_recognizance = Column(Float)
    # 🧠 ML Signal: Column definition for ORM mapping
    # 🧠 ML Signal: Use of SQLAlchemy Column definitions for ORM mapping.
    # ⚠️ SAST Risk (Low): Ensure that the register_schema function is used securely and validate inputs.
    # 🧠 ML Signal: Use of __all__ to define public API of the module.
    #: 投资性房地产
    #
    #: 固定资产
    #
    #: 无形资产
    #
    #: 商誉
    #
    #: 递延所得税资产
    #
    #: 其他资产
    #
    #: 独立账户资产
    fi_capital_in_independent_accounts = Column(Float)
    #: 资产总计
    #
    #: 负债
    #
    #: 短期借款
    #
    #: 同业及其他金融机构存放款项
    #
    #: 拆入资金
    #
    #: 以公允价值计量且其变动计入当期损益的金融负债
    #
    #: 衍生金融负债
    #
    #: 卖出回购金融资产款
    #
    #: 吸收存款
    #
    #: 代理买卖证券款
    #
    #: 应付账款
    #
    #: 预收账款
    fi_advance_from_customers = Column(Float)
    #: 预收保费
    fi_advance_premium = Column(Float)
    #: 应付手续费及佣金
    fi_fees_and_commissions_payable = Column(Float)
    #: 应付分保账款
    fi_dividend_payable_for_reinsurance = Column(Float)
    #: 应付职工薪酬
    #
    #: 应交税费
    #
    #: 应付利息
    #
    #: 预计负债
    #
    #: 应付赔付款
    fi_claims_payable = Column(Float)
    #: 应付保单红利
    fi_policy_holder_dividend_payable = Column(Float)
    #: 保户储金及投资款
    fi_policy_holder_deposits_and_investment_funds = Column(Float)
    #: 保险合同准备金
    fi_contract_reserve = Column(Float)
    #: 长期借款
    #
    #: 应付债券
    #
    #: 递延所得税负债
    #
    #: 其他负债
    #
    #: 独立账户负债
    fi_independent_liability = Column(Float)
    #: 负债合计
    #
    #: 所有者权益(或股东权益)
    #
    #: 股本
    #
    #: 资本公积
    #
    #: 盈余公积
    #
    #: 一般风险准备
    #
    #: 未分配利润
    #
    #: 归属于母公司股东权益总计
    #
    #: 少数股东权益
    #
    #: 股东权益合计
    #
    #: 负债和股东权益总计


class IncomeStatement(FinanceBase, Mixin):
    @classmethod
    def important_cols(cls):
        return [
            "operating_income",
            "investment_income",
            "total_operating_costs",
            "total_profits",
            "sales_costs",
            "managing_costs",
            "financing_costs",
        ]

    __tablename__ = "income_statement"

    provider = Column(String(length=32))
    code = Column(String(length=32))

    report_period = Column(String(length=32))
    report_date = Column(DateTime)

    #: 营业总收入
    #
    #: 营业收入
    operating_income = Column(Float)
    #: 营业总成本
    total_operating_costs = Column(Float)
    #: 营业成本
    operating_costs = Column(Float)
    #: 研发费用
    rd_costs = Column(Float)
    #: 提取保险合同准备金净额
    net_change_in_insurance_contract_reserves = Column(Float)
    #: 营业税金及附加
    business_taxes_and_surcharges = Column(Float)
    #: 销售费用
    sales_costs = Column(Float)
    #: 管理费用
    managing_costs = Column(Float)
    #: 财务费用
    financing_costs = Column(Float)
    #: 资产减值损失
    assets_devaluation = Column(Float)
    #: 其他经营收益
    #
    #: 加: 投资收益
    investment_income = Column(Float)
    #: 其中: 对联营企业和合营企业的投资收益
    investment_income_from_related_enterprise = Column(Float)
    #: 营业利润
    operating_profit = Column(Float)
    #: 加: 营业外收入
    non_operating_income = Column(Float)
    #: 减: 营业外支出
    non_operating_costs = Column(Float)
    #: 其中: 非流动资产处置净损失
    loss_on_disposal_non_current_asset = Column(Float)

    #: 利润总额
    total_profits = Column(Float)
    #: 减: 所得税费用
    tax_expense = Column(Float)
    #: 净利润
    net_profit = Column(Float)
    #: 其中: 归属于母公司股东的净利润
    net_profit_as_parent = Column(Float)
    #: 少数股东损益
    net_profit_as_minority_interest = Column(Float)
    #: 扣除非经常性损益后的净利润
    deducted_net_profit = Column(Float)
    #: 每股收益
    #: 基本每股收益
    eps = Column(Float)
    #: 稀释每股收益
    diluted_eps = Column(Float)
    #: 其他综合收益
    other_comprehensive_income = Column(Float)
    #: 归属于母公司股东的其他综合收益
    other_comprehensive_income_as_parent = Column(Float)
    #: 归属于少数股东的其他综合收益
    other_comprehensive_income_as_minority_interest = Column(Float)
    #: 综合收益总额
    total_comprehensive_income = Column(Float)
    #: 归属于母公司所有者的综合收益总额
    total_comprehensive_income_as_parent = Column(Float)
    #: 归属于少数股东的综合收益总额
    total_comprehensive_income_as_minority_interest = Column(Float)

    #: 银行相关
    #: 利息净收入
    fi_net_interest_income = Column(Float)
    #: 其中:利息收入
    fi_interest_income = Column(Float)
    #: 利息支出
    fi_interest_expenses = Column(Float)
    #: 手续费及佣金净收入
    fi_net_incomes_from_fees_and_commissions = Column(Float)
    #: 其中:手续费及佣金收入
    fi_incomes_from_fees_and_commissions = Column(Float)
    #: 手续费及佣金支出
    fi_expenses_for_fees_and_commissions = Column(Float)
    #: 公允价值变动收益
    fi_income_from_fair_value_change = Column(Float)
    #: 汇兑收益
    fi_income_from_exchange = Column(Float)
    #: 其他业务收入
    fi_other_income = Column(Float)
    #: 业务及管理费
    fi_operate_and_manage_expenses = Column(Float)

    #: 保险相关
    #: 已赚保费
    fi_net_income_from_premium = Column(Float)
    #: 其中:保险业务收入
    fi_income_from_premium = Column(Float)
    #: 分保费收入
    fi_income_from_reinsurance_premium = Column(Float)
    #: 减:分出保费
    fi_reinsurance_premium = Column(Float)
    #: 提取未到期责任准备金
    fi_undue_duty_reserve = Column(Float)
    #: 银行业务利息净收入
    fi_net_income_from_bank_interest = Column(Float)
    #: 其中:银行业务利息收入
    fi_income_from_bank_interest = Column(Float)
    #: 银行业务利息支出
    fi_expenses_for_bank_interest = Column(Float)
    #: 非保险业务手续费及佣金净收入
    fi_net_incomes_from_fees_and_commissions_of_non_insurance = Column(Float)
    #: 非保险业务手续费及佣金收入
    fi_incomes_from_fees_and_commissions_of_non_insurance = Column(Float)
    #: 非保险业务手续费及佣金支出
    fi_expenses_for_fees_and_commissions_of_non_insurance = Column(Float)
    #: 退保金
    fi_insurance_surrender_costs = Column(Float)
    #: 赔付支出
    fi_insurance_claims_expenses = Column(Float)
    #: 减:摊回赔付支出
    fi_amortized_insurance_claims_expenses = Column(Float)
    #: 提取保险责任准备金
    fi_insurance_duty_reserve = Column(Float)
    #: 减:摊回保险责任准备金
    fi_amortized_insurance_duty_reserve = Column(Float)
    #: 保单红利支出
    fi_dividend_expenses_to_insured = Column(Float)
    #: 分保费用
    fi_reinsurance_expenses = Column(Float)
    #: 减:摊回分保费用
    fi_amortized_reinsurance_expenses = Column(Float)
    #: 其他业务成本
    fi_other_op_expenses = Column(Float)

    #: 券商相关
    #: 手续费及佣金净收入
    #
    #: 其中:代理买卖证券业务净收入
    fi_net_incomes_from_trading_agent = Column(Float)
    #: 证券承销业务净收入
    fi_net_incomes_from_underwriting = Column(Float)
    #: 受托客户资产管理业务净收入
    fi_net_incomes_from_customer_asset_management = Column(Float)
    #: 手续费及佣金净收入其他项目
    fi_fees_from_other = Column(Float)
    #: 公允价值变动收益
    #
    #: 其中:可供出售金融资产公允价值变动损益
    fi_income_from_fair_value_change_of_fi_salable = Column(Float)


class CashFlowStatement(FinanceBase, Mixin):
    @classmethod
    def important_cols(cls):
        return [
            "net_op_cash_flows",
            "net_investing_cash_flows",
            "net_financing_cash_flows",
            "cash",
        ]

    __tablename__ = "cash_flow_statement"

    provider = Column(String(length=32))
    code = Column(String(length=32))

    report_period = Column(String(length=32))
    report_date = Column(DateTime)
    #: 经营活动产生的现金流量
    #
    #: 销售商品、提供劳务收到的现金
    cash_from_selling = Column(Float)

    #: 收到的税费返还
    tax_refund = Column(Float)

    #: 收到其他与经营活动有关的现金
    cash_from_other_op = Column(Float)

    #: 经营活动现金流入小计
    total_op_cash_inflows = Column(Float)

    #: 购买商品、接受劳务支付的现金
    cash_to_goods_services = Column(Float)
    #: 支付给职工以及为职工支付的现金
    cash_to_employees = Column(Float)
    #: 支付的各项税费
    taxes_and_surcharges = Column(Float)
    #: 支付其他与经营活动有关的现金
    cash_to_other_related_op = Column(Float)
    #: 经营活动现金流出小计
    total_op_cash_outflows = Column(Float)

    #: 经营活动产生的现金流量净额
    net_op_cash_flows = Column(Float)

    #: 投资活动产生的现金流量

    #: 收回投资收到的现金
    cash_from_disposal_of_investments = Column(Float)
    #: 取得投资收益收到的现金
    cash_from_returns_on_investments = Column(Float)
    #: 处置固定资产、无形资产和其他长期资产收回的现金净额
    cash_from_disposal_fixed_intangible_assets = Column(Float)
    #: 处置子公司及其他营业单位收到的现金净额
    cash_from_disposal_subsidiaries = Column(Float)

    #: 收到其他与投资活动有关的现金
    cash_from_other_investing = Column(Float)

    #: 投资活动现金流入小计
    total_investing_cash_inflows = Column(Float)

    #: 购建固定资产、无形资产和其他长期资产支付的现金
    cash_to_acquire_fixed_intangible_assets = Column(Float)
    #: 投资支付的现金
    cash_to_investments = Column(Float)

    #: 取得子公司及其他营业单位支付的现金净额
    cash_to_acquire_subsidiaries = Column(Float)

    #: 支付其他与投资活动有关的现金
    cash_to_other_investing = Column(Float)

    #: 投资活动现金流出小计
    total_investing_cash_outflows = Column(Float)

    #: 投资活动产生的现金流量净额
    net_investing_cash_flows = Column(Float)

    #: 筹资活动产生的现金流量
    #
    #: 吸收投资收到的现金
    cash_from_accepting_investment = Column(Float)
    #: 子公司吸收少数股东投资收到的现金
    cash_from_subsidiaries_accepting_minority_interest = Column(Float)

    #: 取得借款收到的现金
    cash_from_borrowings = Column(Float)
    #: 发行债券收到的现金
    cash_from_issuing_bonds = Column(Float)
    #: 收到其他与筹资活动有关的现金
    cash_from_other_financing = Column(Float)

    #: 筹资活动现金流入小计
    total_financing_cash_inflows = Column(Float)

    #: 偿还债务支付的现金
    cash_to_repay_borrowings = Column(Float)

    #: 分配股利、利润或偿付利息支付的现金
    cash_to_pay_interest_dividend = Column(Float)

    #: 子公司支付给少数股东的股利、利润
    cash_to_pay_subsidiaries_minority_interest = Column(Float)

    #: 支付其他与筹资活动有关的现金
    cash_to_other_financing = Column(Float)
    #: 筹资活动现金流出小计
    total_financing_cash_outflows = Column(Float)

    #: 筹资活动产生的现金流量净额
    net_financing_cash_flows = Column(Float)
    #: 汇率变动对现金及现金等价物的影响
    foreign_exchange_rate_effect = Column(Float)
    #: 现金及现金等价物净增加额
    net_cash_increase = Column(Float)
    #: 加: 期初现金及现金等价物余额
    cash_at_beginning = Column(Float)
    #: 期末现金及现金等价物余额
    cash = Column(Float)

    #: 银行相关
    #: 客户存款和同业及其他金融机构存放款项净增加额
    fi_deposit_increase = Column(Float)
    #: 向中央银行借款净增加额
    fi_borrow_from_central_bank_increase = Column(Float)
    #: 存放中央银行和同业款项及其他金融机构净减少额
    fi_deposit_in_others_decrease = Column(Float)
    #: 拆入资金及卖出回购金融资产款净增加额
    fi_borrowing_and_sell_repurchase_increase = Column(Float)
    #: 其中:卖出回购金融资产款净增加额
    fi_sell_repurchase_increase = Column(Float)
    #: 拆出资金及买入返售金融资产净减少额
    fi_lending_and_buy_repurchase_decrease = Column(Float)
    #: 其中:拆出资金净减少额
    fi_lending_decrease = Column(Float)
    #: 买入返售金融资产净减少额
    fi_buy_repurchase_decrease = Column(Float)
    #: 收取的利息、手续费及佣金的现金
    fi_cash_from_interest_commission = Column(Float)
    #: 客户贷款及垫款净增加额
    fi_loan_advance_increase = Column(Float)
    #: 存放中央银行和同业及其他金融机构款项净增加额
    fi_deposit_in_others_increase = Column(Float)
    #: 拆出资金及买入返售金融资产净增加额
    fi_lending_and_buy_repurchase_increase = Column(Float)
    #: 其中:拆出资金净增加额
    fi_lending_increase = Column(Float)
    #: 拆入资金及卖出回购金融资产款净减少额
    fi_borrowing_and_sell_repurchase_decrease = Column(Float)
    #: 其中:拆入资金净减少额
    fi_borrowing_decrease = Column(Float)
    #: 卖出回购金融资产净减少额
    fi_sell_repurchase_decrease = Column(Float)
    #: 支付利息、手续费及佣金的现金
    fi_cash_to_interest_commission = Column(Float)
    #: 应收账款净增加额
    fi_account_receivable_increase = Column(Float)
    #: 偿付债券利息支付的现金
    fi_cash_to_pay_interest = Column(Float)

    #: 保险相关
    #: 收到原保险合同保费取得的现金
    fi_cash_from_premium_of_original = Column(Float)
    #: 保户储金及投资款净增加额
    fi_insured_deposit_increase = Column(Float)
    #: 银行及证券业务卖出回购资金净增加额
    fi_bank_broker_sell_repurchase_increase = Column(Float)
    #: 银行及证券业务买入返售资金净减少额
    fi_bank_broker_buy_repurchase_decrease = Column(Float)
    #: 支付原保险合同赔付等款项的现金
    fi_cash_to_insurance_claim = Column(Float)
    #: 支付再保险业务现金净额
    fi_cash_to_reinsurance = Column(Float)
    #: 银行业务及证券业务拆借资金净减少额
    fi_lending_decrease = Column(Float)
    #: 银行业务及证券业务卖出回购资金净减少额
    fi_bank_broker_sell_repurchase_decrease = Column(Float)
    #: 支付保单红利的现金
    fi_cash_to_dividends = Column(Float)
    #: 保户质押贷款净增加额
    fi_insured_pledge_loans_increase = Column(Float)
    #: 收购子公司及其他营业单位支付的现金净额
    fi_cash_to_acquire_subsidiaries = Column(Float)
    #: 处置子公司及其他营业单位流出的现金净额
    fi_cash_to_disposal_subsidiaries = Column(Float)
    #: 支付卖出回购金融资产款现金净额
    fi_cash_to_sell_repurchase = Column(Float)

    #: 券商相关
    #: 拆入资金净增加额
    fi_borrowing_increase = Column(Float)
    #: 代理买卖证券收到的现金净额
    fi_cash_from_trading_agent = Column(Float)
    #: 回购业务资金净增加额
    fi_cash_from_repurchase_increase = Column(Float)
    #: 处置交易性金融资产的净减少额
    fi_disposal_trade_asset_decrease = Column(Float)
    #: 回购业务资金净减少额
    fi_repurchase_decrease = Column(Float)
    #: 代理买卖证券支付的现金净额（净减少额）
    fi_cash_to_agent_trade = Column(Float)


#: 主要财务指标


class FinanceFactor(FinanceBase, Mixin):
    @classmethod
    def important_cols(cls):
        return [
            "basic_eps",
            "total_op_income",
            "net_profit",
            "op_income_growth_yoy",
            "net_profit_growth_yoy",
            "roe",
            "rota",
            "gross_profit_margin",
            "net_margin",
        ]

    __tablename__ = "finance_factor"

    provider = Column(String(length=32))
    code = Column(String(length=32))

    report_period = Column(String(length=32))
    report_date = Column(DateTime)
    #: 每股指标
    #
    #: 基本每股收益(元)
    basic_eps = Column(Float)
    #: 扣非每股收益(元)
    deducted_eps = Column(Float)
    #: 稀释每股收益(元)
    diluted_eps = Column(Float)
    #: 每股净资产(元)
    bps = Column(Float)
    #: 每股资本公积(元)
    capital_reserve_ps = Column(Float)
    #: 每股未分配利润(元)
    undistributed_profit_ps = Column(Float)
    #: 每股经营现金流(元)
    op_cash_flow_ps = Column(Float)
    #: 成长能力指标
    #
    #: 营业总收入(元)
    total_op_income = Column(Float)
    #: 毛利润(元)
    gross_profit = Column(Float)
    #: 归属净利润(元)
    net_profit = Column(Float)
    #: 扣非净利润(元)
    deducted_net_profit = Column(Float)
    #: 营业总收入同比增长
    op_income_growth_yoy = Column(Float)
    #: 归属净利润同比增长
    net_profit_growth_yoy = Column(Float)
    #: 扣非净利润同比增长
    deducted_net_profit_growth_yoy = Column(Float)
    #: 营业总收入滚动环比增长
    op_income_growth_qoq = Column(Float)
    #: 归属净利润滚动环比增长
    net_profit_growth_qoq = Column(Float)
    #: 扣非净利润滚动环比增长
    deducted_net_profit_growth_qoq = Column(Float)
    #: 盈利能力指标
    #
    #: 净资产收益率(加权)
    roe = Column(Float)
    #: 净资产收益率(扣非/加权)
    deducted_roe = Column(Float)
    #: 总资产收益率(加权)
    rota = Column(Float)
    #: 毛利率
    gross_profit_margin = Column(Float)
    #: 净利率
    net_margin = Column(Float)
    #: 收益质量指标
    #
    #: 预收账款/营业收入
    advance_receipts_per_op_income = Column(Float)
    #: 销售净现金流/营业收入
    sales_net_cash_flow_per_op_income = Column(Float)
    #: 经营净现金流/营业收入
    op_net_cash_flow_per_op_income = Column(Float)
    #: 实际税率
    actual_tax_rate = Column(Float)
    #: 财务风险指标
    #
    #: 流动比率
    current_ratio = Column(Float)
    #: 速动比率
    quick_ratio = Column(Float)
    #: 现金流量比率
    cash_flow_ratio = Column(Float)
    #: 资产负债率
    debt_asset_ratio = Column(Float)
    #: 权益乘数
    em = Column(Float)
    #: 产权比率
    equity_ratio = Column(Float)
    #: 营运能力指标(一般企业)
    #
    #: 总资产周转天数(天)
    total_assets_turnover_days = Column(Integer)
    #: 存货周转天数(天)
    inventory_turnover_days = Column(Integer)
    #: 应收账款周转天数(天)
    receivables_turnover_days = Column(Integer)
    #: 总资产周转率(次)
    total_assets_turnover = Column(Float)
    #: 存货周转率(次)
    inventory_turnover = Column(Float)
    #: 应收账款周转率(次)
    receivables_turnover = Column(Float)

    #: 专项指标(银行)
    #
    #: 存款总额
    fi_total_deposit = Column(Float)
    #: 贷款总额
    fi_total_loan = Column(Float)
    #: 存贷款比例
    fi_loan_deposit_ratio = Column(Float)
    #: 资本充足率
    fi_capital_adequacy_ratio = Column(Float)
    #: 核心资本充足率
    fi_core_capital_adequacy_ratio = Column(Float)
    #: 不良贷款率
    fi_npl_ratio = Column(Float)
    #: 不良贷款拨备覆盖率
    fi_npl_provision_coverage = Column(Float)
    #: 资本净额
    fi_net_capital = Column(Float)
    #: 专项指标(保险)
    #
    #: 总投资收益率
    insurance_roi = Column(Float)
    #: 净投资收益率
    insurance_net_investment_yield = Column(Float)
    #: 已赚保费
    insurance_earned_premium = Column(Float)
    #: 赔付支出
    insurance_payout = Column(Float)
    #: 退保率
    insurance_surrender_rate = Column(Float)
    #: 偿付能力充足率
    insurance_solvency_adequacy_ratio = Column(Float)
    #: 专项指标(券商)
    #
    #: 净资本
    broker_net_capital = Column(Float)
    #: 净资产
    broker_net_assets = Column(Float)
    #: 净资本/净资产
    broker_net_capital_assets_ratio = Column(Float)
    #: 自营固定收益类证券规模/净资本
    broker_self_operated_fixed_income_securities_net_capital_ratio = Column(Float)


register_schema(
    providers=["eastmoney"],
    db_name="finance",
    schema_base=FinanceBase,
    entity_type="stock",
)


# the __all__ is generated
__all__ = ["BalanceSheet", "IncomeStatement", "CashFlowStatement", "FinanceFactor"]
