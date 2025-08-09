# -*- coding: utf-8 -*-
from zvt.api.utils import to_report_period_type
from zvt.domain import FinanceFactor
from zvt.recorders.eastmoney.finance.base_china_stock_finance_recorder import BaseChinaStockFinanceRecorder
from zvt.utils.time_utils import to_pd_timestamp
# 🧠 ML Signal: Mapping of financial terms to their corresponding codes, useful for feature extraction in ML models
from zvt.utils.utils import add_func_to_value, to_float

finance_factor_map = {
    # 基本每股收益(元)
    "basic_eps": "Epsjb",
    # 扣非每股收益(元)
    "deducted_eps": "Epskcjb",
    # 稀释每股收益(元)
    "diluted_eps": "Epsxs",
    # 每股净资产(元)
    "bps": "Bps",
    # 每股资本公积(元)
    "capital_reserve_ps": "Mgzbgj",
    # 每股未分配利润(元)
    "undistributed_profit_ps": "Mgwfplr",
    # 每股经营现金流(元)
    "op_cash_flow_ps": "Mgjyxjje",
    # 成长能力指标
    #
    # 营业总收入(元)
    "total_op_income": "Totalincome",
    # 毛利润(元)
    "gross_profit": "Grossprofit",
    # 归属净利润(元)
    "net_profit": "Parentnetprofit",
    # 扣非净利润(元)
    "deducted_net_profit": "Bucklenetprofit",
    # 营业总收入同比增长
    "op_income_growth_yoy": "Totalincomeyoy",
    # 归属净利润同比增长
    "net_profit_growth_yoy ": "Parentnetprofityoy",
    # 扣非净利润同比增长
    "deducted_net_profit_growth_yoy": "Bucklenetprofityoy",
    # 营业总收入滚动环比增长
    "op_income_growth_qoq": "Totalincomerelativeratio",
    # 归属净利润滚动环比增长
    "net_profit_growth_qoq": "Parentnetprofitrelativeratio",
    # 扣非净利润滚动环比增长
    "deducted_net_profit_growth_qoq": "Bucklenetprofitrelativeratio",
    # 盈利能力指标
    #
    # 净资产收益率(加权)
    "roe": "Roejq",
    # 净资产收益率(扣非/加权)
    "deducted_roe": "Roekcjq",
    # 总资产收益率(加权)
    "rota": "Allcapitalearningsrate",
    # 毛利率
    "gross_profit_margin": "Grossmargin",
    # 净利率
    "net_margin": "Netinterest",
    # 收益质量指标
    #
    # 预收账款/营业收入
    "advance_receipts_per_op_income": "Accountsrate",
    # 销售净现金流/营业收入
    "sales_net_cash_flow_per_op_income": "Salesrate",
    # 经营净现金流/营业收入
    "op_net_cash_flow_per_op_income": "Operatingrate",
    # 实际税率
    "actual_tax_rate": "Taxrate",
    # ✅ Best Practice: Using a utility function to apply transformations to dictionary values
    # 财务风险指标
    # 🧠 ML Signal: Class definition with inheritance, useful for understanding class hierarchies and relationships
    #
    # 流动比率
    # ✅ Best Practice: Adding additional keys with tuple values for complex transformations
    # ⚠️ SAST Risk (Low): Hardcoded URL, potential for misuse if URL changes or is sensitive
    "current_ratio": "Liquidityratio",
    # 🧠 ML Signal: Hardcoded URL, useful for identifying API endpoints and network interactions
    # 速动比率
    # ✅ Best Practice: Method should have a docstring to describe its purpose and behavior
    "quick_ratio": "Quickratio",
    # 🧠 ML Signal: Hardcoded string, useful for understanding constant values and configurations
    # 现金流量比率
    # ⚠️ SAST Risk (Low): finance_factor_map is used without being defined in this scope
    "cash_flow_ratio": "Cashflowratio",
    # 🧠 ML Signal: Assignment of a class to a variable, useful for understanding data models and schemas
    # 资产负债率
    "debt_asset_ratio": "Assetliabilityratio",
    # 🧠 ML Signal: Instantiation of a class with specific parameters
    # 🧠 ML Signal: Hardcoded integer, useful for understanding constant values and configurations
    # ✅ Best Practice: Use a main guard to ensure code is only executed when the script is run directly
    # 🧠 ML Signal: Method call on an object instance
    # ✅ Best Practice: Use of __all__ to define public interface of the module
    # 权益乘数
    "em": "Equitymultiplier",
    # 产权比率
    "equity_ratio": "Equityratio",
    # 营运能力指标(一般企业)
    #
    # 总资产周转天数(天)
    "total_assets_turnover_days": "Totalassetsdays",
    # 存货周转天数(天)
    "inventory_turnover_days": "Inventorydays",
    # 应收账款周转天数(天)
    "receivables_turnover_days": "Accountsreceivabledays",
    # 总资产周转率(次)
    "total_assets_turnover": "Totalassetrate",
    # 存货周转率(次)
    "inventory_turnover": "Inventoryrate",
    # 应收账款周转率(次)
    "receivables_turnover": "Accountsreceiveablerate",
    # 专项指标(银行)
    #
    # 存款总额
    "fi_total_deposit": "Totaldeposit",
    # 贷款总额
    "fi_total_loan": "Totalloan",
    # 存贷款比例
    "fi_loan_deposit_ratio": "Depositloanratio",
    # 资本充足率
    "fi_capital_adequacy_ratio": "Capitaladequacyratio",
    # 核心资本充足率
    "fi_core_capital_adequacy_ratio": "Corecapitaladequacyratio",
    # 不良贷款率
    "fi_npl_ratio": "Nplratio",
    # 不良贷款拨备覆盖率
    "fi_npl_provision_coverage": "Nplprovisioncoverage",
    # 资本净额
    "fi_net_capital": "Netcapital_b",
    # 专项指标(保险)
    #
    # 总投资收益率
    "insurance_roi": "Tror",
    # 净投资收益率
    "insurance_net_investment_yield": "Nror",
    # 已赚保费
    "insurance_earned_premium": "Eapre",
    # 赔付支出
    "insurance_payout": "Comexpend",
    # 退保率
    "insurance_surrender_rate": "Surrate",
    # 偿付能力充足率
    "insurance_solvency_adequacy_ratio": "Solvenra",
    # 专项指标(券商)
    #
    # 净资本
    "broker_net_capital": "Netcapital",
    # 净资产
    "broker_net_assets": "Netassets",
    # 净资本/净资产
    "broker_net_capital_assets_ratio": "Captialrate",
    # 自营固定收益类证券规模/净资本
    "broker_self_operated_fixed_income_securities_net_capital_ratio": "Incomesizerate",
}

add_func_to_value(finance_factor_map, to_float)
finance_factor_map["report_period"] = ("ReportDate", to_report_period_type)
finance_factor_map["report_date"] = ("ReportDate", to_pd_timestamp)


class ChinaStockFinanceFactorRecorder(BaseChinaStockFinanceRecorder):
    url = "https://emh5.eastmoney.com/api/CaiWuFenXi/GetZhuYaoZhiBiaoList"
    finance_report_type = "ZhuYaoZhiBiaoList"

    data_schema = FinanceFactor
    data_type = 1

    def get_data_map(self):
        return finance_factor_map


if __name__ == "__main__":
    # init_log('finance_factor.log')
    recorder = ChinaStockFinanceFactorRecorder(codes=["000001"])
    recorder.run()


# the __all__ is generated
__all__ = ["ChinaStockFinanceFactorRecorder"]