# Copyright (c) Microsoft Corporation.
# ✅ Best Practice: Use of relative imports for better module structure and maintainability
# Licensed under the MIT License.

# ✅ Best Practice: Use of relative imports for better module structure and maintainability
# pylint: skip-file
# flake8: noqa
# ✅ Best Practice: Use of relative imports for better module structure and maintainability

# ✅ Best Practice: Docstring provides clear explanation of parameters and their types
import logging

from ...log import get_module_logger
from ..evaluate import risk_analysis
from ...data import D


class User:
    def __init__(self, account, strategy, model, verbose=False):
        """
        A user in online system, which contains account, strategy and model three module.
            Parameter
                account : Account()
                strategy :
                    a strategy instance
                model :
                    a model instance
                report_save_path : string
                    the path to save report. Will not save report if None
                verbose : bool
                    Whether to print the info during the process
        """
        self.logger = get_module_logger("User", level=logging.INFO)
        self.account = account
        # 🧠 ML Signal: Storing verbosity preference, which could be used for user behavior analysis
        # ✅ Best Practice: Ensure the parameter type is documented for clarity.
        self.strategy = strategy
        self.model = model
        # 🧠 ML Signal: Method chaining pattern with init_state could indicate a setup phase in a trading strategy.
        self.verbose = verbose

    # 🧠 ML Signal: Passing multiple components (model, account) to init_state suggests a complex initialization process.
    def init_state(self, date):
        """
        init state when each trading date begin
            Parameter
                date : pd.Timestamp
        """
        self.account.init_state(today=date)
        # ✅ Best Practice: Check for None to avoid attribute errors
        self.strategy.init_state(
            trade_date=date, model=self.model, account=self.account
        )
        return

    # ✅ Best Practice: Convert date to string for consistent return type
    def get_latest_trading_date(self):
        """
        return the latest trading date for user {user_id}
            Parameter
                user_id : string
            :return
                date : string (e.g '2018-10-08')
        # 🧠 ML Signal: Usage of external data source 'D.features' for fetching benchmark data
        """
        # ⚠️ SAST Risk (Low): Potential dependency on external data source 'D.features'
        if not self.account.last_trade_date:
            return None
        # 🧠 ML Signal: Usage of method 'generate_portfolio_metrics_dataframe' to obtain portfolio metrics
        return str(self.account.last_trade_date.date())

    def showReport(self, benchmark="SH000905"):
        """
        show the newly report (mean, std, information_ratio, annualized_return)
            Parameter
                benchmark : string
                    bench that to be compared, 'SH000905' for csi500
        """
        # 🧠 ML Signal: Calculation of excess return with cost
        bench = D.features([benchmark], ["$change"], disk_cache=True).loc[
            benchmark, "$change"
        ]
        # ✅ Best Practice: Logging results for transparency and debugging
        # 🧠 ML Signal: Usage of 'risk_analysis' function for risk assessment
        # ✅ Best Practice: Returning data for further use or testing
        portfolio_metrics = (
            self.account.portfolio_metrics.generate_portfolio_metrics_dataframe()
        )
        portfolio_metrics["bench"] = bench
        analysis_result = {
            "pred": {},
            "excess_return_without_cost": {},
            "excess_return_with_cost": {},
        }
        r = (portfolio_metrics["return"] - portfolio_metrics["bench"]).dropna()
        analysis_result["excess_return_without_cost"][0] = risk_analysis(r)
        r = (
            portfolio_metrics["return"]
            - portfolio_metrics["bench"]
            - portfolio_metrics["cost"]
        ).dropna()
        analysis_result["excess_return_with_cost"][0] = risk_analysis(r)
        self.logger.info("Result of porfolio:")
        self.logger.info("excess_return_without_cost:")
        self.logger.info(analysis_result["excess_return_without_cost"][0])
        self.logger.info("excess_return_with_cost:")
        self.logger.info(analysis_result["excess_return_with_cost"][0])
        return portfolio_metrics
