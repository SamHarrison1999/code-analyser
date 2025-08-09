# -*- coding: utf-8 -*-
from zvt.api.utils import to_report_period_type
from zvt.domain import CashFlowStatement
from zvt.recorders.eastmoney.finance.base_china_stock_finance_recorder import BaseChinaStockFinanceRecorder
from zvt.utils.time_utils import to_pd_timestamp
# 🧠 ML Signal: Mapping of cash flow terms to their corresponding codes, useful for feature extraction in ML models
from zvt.utils.utils import add_func_to_value, first_item_to_float

cash_flow_map = {
    # 经营活动产生的现金流量
    #
    # 销售商品、提供劳务收到的现金
    "cash_from_selling": "Salegoodsservicerec",
    # 收到的税费返还
    "tax_refund": "Taxreturnrec",
    # 收到其他与经营活动有关的现金
    "cash_from_other_op": "Otheroperaterec",
    # 经营活动现金流入小计
    "total_op_cash_inflows": "Sumoperateflowin",
    # 购买商品、接受劳务支付的现金
    "cash_to_goods_services": "Buygoodsservicepay",
    # 支付给职工以及为职工支付的现金
    "cash_to_employees": "Employeepay",
    # 支付的各项税费
    "taxes_and_surcharges": "Taxpay",
    # 支付其他与经营活动有关的现金
    "cash_to_other_related_op": "Otheroperatepay",
    # 经营活动现金流出小计
    "total_op_cash_outflows": "Sumoperateflowout",
    # 经营活动产生的现金流量净额
    "net_op_cash_flows": "Netoperatecashflow",
    # 投资活动产生的现金流量
    # 收回投资收到的现金
    "cash_from_disposal_of_investments": "Disposalinvrec",
    # 取得投资收益收到的现金
    "cash_from_returns_on_investments": "Invincomerec",
    # 处置固定资产、无形资产和其他长期资产收回的现金净额
    "cash_from_disposal_fixed_intangible_assets": "Dispfilassetrec",
    # 处置子公司及其他营业单位收到的现金净额
    "cash_from_disposal_subsidiaries": "Dispsubsidiaryrec",
    # 收到其他与投资活动有关的现金
    "cash_from_other_investing": "Otherinvrec",
    # 投资活动现金流入小计
    "total_investing_cash_inflows": "Suminvflowin",
    # 购建固定资产、无形资产和其他长期资产支付的现金
    "cash_to_acquire_fixed_intangible_assets": "Buyfilassetpay",
    # 投资支付的现金
    "cash_to_investments": "Invpay",
    # 取得子公司及其他营业单位支付的现金净额
    "cash_to_acquire_subsidiaries": "Getsubsidiarypay",
    # 支付其他与投资活动有关的现金
    "cash_to_other_investing": "Otherinvpay",
    # 投资活动现金流出小计
    "total_investing_cash_outflows": "Suminvflowout",
    # 投资活动产生的现金流量净额
    "net_investing_cash_flows": "Netinvcashflow",
    # 筹资活动产生的现金流量
    #
    # 吸收投资收到的现金
    "cash_from_accepting_investment": "Acceptinvrec",
    # 子公司吸收少数股东投资收到的现金
    "cash_from_subsidiaries_accepting_minority_interest": "Subsidiaryaccept",
    # 取得借款收到的现金
    "cash_from_borrowings": "Loanrec",
    # 发行债券收到的现金
    "cash_from_issuing_bonds": "Issuebondrec",
    # 收到其他与筹资活动有关的现金
    "cash_from_other_financing": "Otherfinarec",
    # 筹资活动现金流入小计
    "total_financing_cash_inflows": "Sumfinaflowin",
    # 偿还债务支付的现金
    "cash_to_repay_borrowings": "Repaydebtpay",
    # 分配股利、利润或偿付利息支付的现金
    "cash_to_pay_interest_dividend": "Diviprofitorintpay",
    # 子公司支付给少数股东的股利、利润
    "cash_to_pay_subsidiaries_minority_interest": "Subsidiarypay",
    # 支付其他与筹资活动有关的现金
    "cash_to_other_financing": "Otherfinapay",
    # 筹资活动现金流出小计
    "total_financing_cash_outflows": "Sumfinaflowout",
    # 筹资活动产生的现金流量净额
    "net_financing_cash_flows": "Netfinacashflow",
    # 汇率变动对现金及现金等价物的影响
    "foreign_exchange_rate_effect": "Effectexchangerate",
    # 现金及现金等价物净增加额
    "net_cash_increase": "Nicashequi",
    # 加: 期初现金及现金等价物余额
    # ✅ Best Practice: Using a utility function to apply transformations to dictionary values
    # 🧠 ML Signal: Class definition for a specific financial data recorder
    "cash_at_beginning": "Cashequibeginning",
    # 期末现金及现金等价物余额
    # 🧠 ML Signal: Specifies the data schema used for the recorder
    "cash": "Cashequiending",
    # 🧠 ML Signal: Adding additional fields to the cash flow map for report period and date, useful for temporal analysis
    # 银行相关
    # 🧠 ML Signal: URL for fetching financial data
    # 客户存款和同业及其他金融机构存放款项净增加额
    # ⚠️ SAST Risk (Low): The function get_data_map accesses cash_flow_map, which is not defined within the function or passed as a parameter.
    "fi_deposit_increase": "Nideposit",
    # 🧠 ML Signal: Type of financial report being recorded
    # 向中央银行借款净增加额
    "fi_borrow_from_central_bank_increase": "Niborrowfromcbank",
    # 🧠 ML Signal: Indicates a specific data type used in the recorder
    # ✅ Best Practice: Use the standard Python idiom for checking if a script is run as the main program.
    # 存放中央银行和同业款项及其他金融机构净减少额
    "fi_deposit_in_others_decrease": "Nddepositincbankfi",
    # 🧠 ML Signal: Instantiating a class with specific parameters can indicate typical usage patterns.
    # 🧠 ML Signal: Calling the run method on an object can indicate a common entry point for execution.
    # ✅ Best Practice: Defining __all__ specifies the public API of the module, which improves code readability and maintainability.
    # 拆入资金及卖出回购金融资产款净增加额
    "fi_borrowing_and_sell_repurchase_increase": "Niborrowsellbuyback",
    # 其中:卖出回购金融资产款净增加额
    "fi_sell_repurchase_increase": "Nisellbuybackfasset",
    # 拆出资金及买入返售金融资产净减少额
    "fi_lending_and_buy_repurchase_decrease": "Ndlendbuysellback",
    # 其中:拆出资金净减少额
    "fi_lending_decrease": "Ndlendfund",
    # 买入返售金融资产净减少额
    "fi_buy_repurchase_decrease": "Ndbuysellbackfasset",
    # 收取的利息、手续费及佣金的现金
    "fi_cash_from_interest_commission": "Intandcommrec",
    # 客户贷款及垫款净增加额
    "fi_loan_advance_increase": "Niloanadvances",
    # 存放中央银行和同业及其他金融机构款项净增加额
    "fi_deposit_in_others_increase": "Nidepositincbankfi",
    # 拆出资金及买入返售金融资产净增加额
    "fi_lending_and_buy_repurchase_increase": "Nilendsellbuyback",
    # 其中:拆出资金净增加额
    "fi_lending_increase": "Nilendfund",
    # 拆入资金及卖出回购金融资产款净减少额
    "fi_borrowing_and_sell_repurchase_decrease": "Ndborrowsellbuyback",
    # 其中:拆入资金净减少额
    "fi_borrowing_decrease": "Ndborrowfund",
    # 卖出回购金融资产净减少额
    "fi_sell_repurchase_decrease": "Ndsellbuybackfasset",
    # 支付利息、手续费及佣金的现金
    "fi_cash_to_interest_commission": "Intandcommpay",
    # 应收账款净增加额
    "fi_account_receivable_increase": "Niaccountrec",
    # 偿付债券利息支付的现金
    "fi_cash_to_pay_interest": "Bondintpay",
    # 保险相关
    # 收到原保险合同保费取得的现金
    "fi_cash_from_premium_of_original": "Premiumrec",
    # 保户储金及投资款净增加额
    "fi_insured_deposit_increase": "Niinsureddepositinv",
    # 银行及证券业务卖出回购资金净增加额
    "fi_bank_broker_sell_repurchase_increase": "Nisellbuyback",
    # 银行及证券业务买入返售资金净减少额
    "fi_bank_broker_buy_repurchase_decrease": "Ndbuysellback",
    # 支付原保险合同赔付等款项的现金
    "fi_cash_to_insurance_claim": "Indemnitypay",
    # 支付再保险业务现金净额
    "fi_cash_to_reinsurance": "Netripay",
    # 银行业务及证券业务拆借资金净减少额
    "fi_lending_decrease": "Ndlendfund",
    # 银行业务及证券业务卖出回购资金净减少额
    "fi_bank_broker_sell_repurchase_decrease": "Ndsellbuyback",
    # 支付保单红利的现金
    "fi_cash_to_dividends": "Divipay",
    # 保户质押贷款净增加额
    "fi_insured_pledge_loans_increase": "Niinsuredpledgeloan",
    # 收购子公司及其他营业单位支付的现金净额
    "fi_cash_to_acquire_subsidiaries": "Buysubsidiarypay",
    # 处置子公司及其他营业单位流出的现金净额
    "fi_cash_to_disposal_subsidiaries": "Dispsubsidiarypay",
    # 支付卖出回购金融资产款现金净额
    "fi_cash_to_sell_repurchase": "Netsellbuybackfassetpay",
    # 券商相关
    # 拆入资金净增加额
    "fi_borrowing_increase": "Niborrowfund",
    # 代理买卖证券收到的现金净额
    "fi_cash_from_trading_agent": "Agenttradesecurityrec",
    # 回购业务资金净增加额
    "fi_cash_from_repurchase_increase": "Nibuybackfund",
    # 处置交易性金融资产的净减少额
    "fi_disposal_trade_asset_decrease": "Nddisptradefasset",
    # 回购业务资金净减少额
    "fi_repurchase_decrease": "Ndbuybackfund",
    # 代理买卖证券支付的现金净额（净减少额）
    "fi_cash_to_agent_trade": "Agenttradesecuritypay",
}

add_func_to_value(cash_flow_map, first_item_to_float)
cash_flow_map["report_period"] = ("ReportDate", to_report_period_type)
cash_flow_map["report_date"] = ("ReportDate", to_pd_timestamp)


class ChinaStockCashFlowRecorder(BaseChinaStockFinanceRecorder):
    data_schema = CashFlowStatement

    url = "https://emh5.eastmoney.com/api/CaiWuFenXi/GetXianJinLiuLiangBiaoList"
    finance_report_type = "XianJinLiuLiangBiaoList"
    data_type = 4

    def get_data_map(self):
        return cash_flow_map


if __name__ == "__main__":
    # init_log('cash_flow.log')
    recorder = ChinaStockCashFlowRecorder(codes=["002572"])
    recorder.run()


# the __all__ is generated
__all__ = ["ChinaStockCashFlowRecorder"]