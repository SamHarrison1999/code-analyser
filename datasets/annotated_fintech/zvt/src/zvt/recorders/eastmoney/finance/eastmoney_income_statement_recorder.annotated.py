# -*- coding: utf-8 -*-
from zvt.api.utils import to_report_period_type
from zvt.domain import IncomeStatement
from zvt.recorders.eastmoney.finance.base_china_stock_finance_recorder import (
    BaseChinaStockFinanceRecorder,
)
from zvt.utils.time_utils import to_pd_timestamp

# 🧠 ML Signal: Dictionary mapping for financial terms, useful for feature extraction in ML models
from zvt.utils.utils import add_func_to_value, first_item_to_float

income_statement_map = {
    # 营业总收入
    #
    # 营业收入
    "operating_income": "Operatereve",
    # 营业总成本
    "total_operating_costs": "Totaloperateexp",
    # 营业成本
    "operating_costs": "Operateexp",
    # 研发费用
    "rd_costs": "Rdexp",
    # 提取保险合同准备金净额
    "net_change_in_insurance_contract_reserves": "Netcontactreserve",
    # 营业税金及附加
    "business_taxes_and_surcharges": "Operatetax",
    # 销售费用
    "sales_costs": "Saleexp",
    # 管理费用
    "managing_costs": "Manageexp",
    # 财务费用
    "financing_costs": "Financeexp",
    # 资产减值损失
    "assets_devaluation": "Assetdevalueloss",
    # 其他经营收益
    #
    # 加: 投资收益
    "investment_income": "Investincome",
    # 其中: 对联营企业和合营企业的投资收益
    "investment_income_from_related_enterprise": "Investjointincome",
    # 营业利润
    "operating_profit": "Operateprofit",
    # 加: 营业外收入
    "non_operating_income": "Nonoperatereve",
    # 减: 营业外支出
    "non_operating_costs": "Nonoperateexp",
    # 其中: 非流动资产处置净损失
    "loss_on_disposal_non_current_asset": "Nonlassetnetloss",
    # 利润总额
    "total_profits": "Sumprofit",
    # 减: 所得税费用
    "tax_expense": "Incometax",
    # 净利润
    "net_profit": "Netprofit",
    # 其中: 归属于母公司股东的净利润
    "net_profit_as_parent": "Parentnetprofit",
    # 少数股东损益
    "net_profit_as_minority_interest": "Minorityincome",
    # 扣除非经常性损益后的净利润
    "deducted_net_profit": "Kcfjcxsyjlr",
    # 每股收益
    # 基本每股收益
    "eps": "Basiceps",
    # 稀释每股收益
    "diluted_eps": "Dilutedeps",
    # 其他综合收益
    "other_comprehensive_income": "Othercincome",
    # 归属于母公司股东的其他综合收益
    "other_comprehensive_income_as_parent": "Parentothercincome",
    # 归属于少数股东的其他综合收益
    "other_comprehensive_income_as_minority_interest": "Minorityothercincome",
    # 综合收益总额
    "total_comprehensive_income": "Sumcincome",
    # 归属于母公司所有者的综合收益总额
    "total_comprehensive_income_as_parent": "Parentcincome",
    # 归属于少数股东的综合收益总额
    "total_comprehensive_income_as_minority_interest": "Minoritycincome",
    # 银行相关
    # 利息净收入
    # ✅ Best Practice: Using a utility function to apply transformations to dictionary values
    "fi_net_interest_income": "Intnreve",
    # 🧠 ML Signal: Class definition with inheritance, useful for understanding class hierarchies and relationships
    # 其中:利息收入
    "fi_interest_income": "Intreve",
    # ✅ Best Practice: Adding additional keys with transformation functions for specific data processing
    # 🧠 ML Signal: Static URL assignment, useful for detecting hardcoded endpoints
    # 利息支出
    "fi_interest_expenses": "Intexp",
    # ⚠️ SAST Risk (Low): Hardcoded URL can lead to inflexibility and potential security risks if the endpoint changes or is deprecated
    # 🧠 ML Signal: Method definition without parameters
    # 手续费及佣金净收入
    "fi_net_incomes_from_fees_and_commissions": "Commnreve",
    # 🧠 ML Signal: Static string assignment, useful for detecting hardcoded values
    # 🧠 ML Signal: Returning a variable from a method
    # 其中:手续费及佣金收入
    "fi_incomes_from_fees_and_commissions": "Commreve",
    # 🧠 ML Signal: Static integer assignment, useful for detecting hardcoded values
    # 手续费及佣金支出
    # 🧠 ML Signal: Instantiating a class with specific parameters
    # 🧠 ML Signal: Calling a method on an object
    # ✅ Best Practice: Defining __all__ to specify public API of the module
    # 🧠 ML Signal: Checking if the script is run as the main program
    "fi_expenses_for_fees_and_commissions": "Commexp",
    # 公允价值变动收益
    "fi_income_from_fair_value_change": "Fvalueincome",
    # 汇兑收益
    "fi_income_from_exchange": "Exchangeincome",
    # 其他业务收入
    "fi_other_income": "Otherreve",
    # 业务及管理费
    "fi_operate_and_manage_expenses": "Operatemanageexp",
    # 保险相关
    # 已赚保费
    "fi_net_income_from_premium": "Premiumearned",
    # 其中:保险业务收入
    "fi_income_from_premium": "Insurreve",
    # 分保费收入
    "fi_income_from_reinsurance_premium": "Rireve",
    # 减:分出保费
    "fi_reinsurance_premium": "Ripremium",
    # 提取未到期责任准备金
    "fi_undue_duty_reserve": "Unduereserve",
    # 银行业务利息净收入
    "fi_net_income_from_bank_interest": "Bankintnreve",
    # 其中:银行业务利息收入
    "fi_income_from_bank_interest": "Bankintreve",
    # 银行业务利息支出
    "fi_expenses_for_bank_interest": "Bankintexp",
    # 非保险业务手续费及佣金净收入
    "fi_net_incomes_from_fees_and_commissions_of_non_insurance": "Ninsurcommnreve",
    # 非保险业务手续费及佣金收入
    "fi_incomes_from_fees_and_commissions_of_non_insurance": "Ninsurcommreve",
    # 非保险业务手续费及佣金支出
    "fi_expenses_for_fees_and_commissions_of_non_insurance": "Ninsurcommexp",
    # 退保金
    "fi_insurance_surrender_costs": "Surrenderpremium",
    # 赔付支出
    "fi_insurance_claims_expenses": "Indemnityexp",
    # 减:摊回赔付支出
    "fi_amortized_insurance_claims_expenses": "Amortiseindemnityexp",
    # 提取保险责任准备金
    "fi_insurance_duty_reserve": "Dutyreserve",
    # 减:摊回保险责任准备金
    "fi_amortized_insurance_duty_reserve": "Amortisedutyreserve",
    # 保单红利支出
    "fi_dividend_expenses_to_insured": "Policydiviexp",
    # 分保费用
    "fi_reinsurance_expenses": "Riexp",
    # 减:摊回分保费用
    "fi_amortized_reinsurance_expenses": "Amortiseriexp",
    # 其他业务成本
    "fi_other_op_expenses": "Otherexp",
    # 券商相关
    # 手续费及佣金净收入
    #
    # 其中:代理买卖证券业务净收入
    "fi_net_incomes_from_trading_agent": "Agenttradesecurity",
    # 证券承销业务净收入
    "fi_net_incomes_from_underwriting": "Securityuw",
    # 受托客户资产管理业务净收入
    "fi_net_incomes_from_customer_asset_management": "Clientassetmanage",
    # 手续费及佣金净收入其他项目
    "fi_fees_from_other": "Commnreveother",
    # 公允价值变动收益
    #
    # 其中:可供出售金融资产公允价值变动损益
    "fi_income_from_fair_value_change_of_fi_salable": "Fvalueosalable",
}

add_func_to_value(income_statement_map, first_item_to_float)
income_statement_map["report_period"] = ("ReportDate", to_report_period_type)
income_statement_map["report_date"] = ("ReportDate", to_pd_timestamp)


class ChinaStockIncomeStatementRecorder(BaseChinaStockFinanceRecorder):
    data_schema = IncomeStatement

    url = "https://emh5.eastmoney.com/api/CaiWuFenXi/GetLiRunBiaoList"
    finance_report_type = "LiRunBiaoList"

    data_type = 2

    def get_data_map(self):
        return income_statement_map


if __name__ == "__main__":
    # init_log('income_statement.log')
    recorder = ChinaStockIncomeStatementRecorder(codes=["002572"])
    recorder.run()


# the __all__ is generated
__all__ = ["ChinaStockIncomeStatementRecorder"]
