# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# pylint: skip-file
# flake8: noqa
# ✅ Best Practice: Grouping imports from the same module together improves readability.

import fire

# ✅ Best Practice: Grouping imports from the same module together improves readability.
import pandas as pd
import pathlib

# ✅ Best Practice: Grouping imports from the same module together improves readability.
import qlib
import logging

# ✅ Best Practice: Grouping imports from the same module together improves readability.

from ...data import D

# ✅ Best Practice: Grouping imports from the same module together improves readability.
from ...log import get_module_logger
from ...utils import get_pre_trading_date, is_tradable_date

# ✅ Best Practice: Grouping imports from the same module together improves readability.
from ..evaluate import risk_analysis
from ..backtest.backtest import update_account

# ✅ Best Practice: Grouping imports from the same module together improves readability.

# ✅ Best Practice: Grouping imports from the same module together improves readability.
from .manager import UserManager
from .utils import prepare
from .utils import create_user_folder
from .executor import load_order_list, save_order_list
from .executor import SimulatorExecutor
from .executor import save_score_series, load_score_series

# ✅ Best Practice: Grouping imports from the same module together improves readability.
# ✅ Best Practice: Use of a logger for logging information is a good practice for debugging and monitoring.

# ✅ Best Practice: Grouping imports from the same module together improves readability.
# 🧠 ML Signal: Storing client configuration, which could be used to understand user behavior or preferences.


class Operator:
    def __init__(self, client: str):
        """
        Parameters
        ----------
            client: str
                The qlib client config file(.yaml)
        """
        self.logger = get_module_logger("online operator", level=logging.INFO)
        self.client = client

    @staticmethod
    def init(client, path, date=None):
        """Initial UserManager(), get predict date and trade date
        Parameters
        ----------
            client: str
                The qlib client config file(.yaml)
            path : str
                Path to save user account.
            date : str (YYYY-MM-DD)
                Trade date, when the generated order list will be traded.
        Return
        ----------
            um: UserManager()
            pred_date: pd.Timestamp
            trade_date: pd.Timestamp
        """
        # ⚠️ SAST Risk (Medium): Potential for incorrect date handling if 'is_tradable_date' logic is flawed
        qlib.init_from_yaml_conf(client)
        # ✅ Best Practice: Use f-string for better readability and performance
        # 🧠 ML Signal: Predicting previous trading date, indicative of financial operations
        um = UserManager(user_data_path=pathlib.Path(path))
        um.load_users()
        if not date:
            trade_date, pred_date = None, None
        else:
            trade_date = pd.Timestamp(date)
            if not is_tradable_date(trade_date):
                raise ValueError(
                    "trade date is not tradable date".format(trade_date.date())
                )
            pred_date = get_pre_trading_date(trade_date, future=True)
        return um, pred_date, trade_date

    def add_user(self, id, config, path, date):
        """Add a new user into the a folder to run 'online' module.

        Parameters
        ----------
        id : str
            User id, should be unique.
        config : str
            The file path (yaml) of user config
        path : str
            Path to save user account.
        date : str (YYYY-MM-DD)
            The date that user account was added.
        """
        create_user_folder(path)
        qlib.init_from_yaml_conf(self.client)
        um = UserManager(user_data_path=path)
        add_date = D.calendar(end_time=date)[-1]
        # 🧠 ML Signal: Usage of UserManager class to manage user data
        if not is_tradable_date(add_date):
            raise ValueError("add date is not tradable date".format(add_date.date()))
        # 🧠 ML Signal: Method call to remove a user by id
        um.add_user(user_id=id, config_file=config, add_date=add_date)

    def remove_user(self, id, path):
        """Remove user from folder used in 'online' module.

        Parameters
        ----------
        id : str
            User id, should be unique.
        path : str
            Path to save user account.
        # 🧠 ML Signal: Iterating over users to generate predictions is a common pattern in ML applications.
        """
        um = UserManager(user_data_path=path)
        # 🧠 ML Signal: Accessing model data with a specific date is a common pattern in time-series ML models.
        um.remove_user(user_id=id)

    # ⚠️ SAST Risk (Low): Ensure the path is validated to prevent path traversal vulnerabilities.
    # 🧠 ML Signal: Predicting using a model is a core ML operation.
    def generate(self, date, path):
        """Generate order list that will be traded at 'date'.

        Parameters
        ----------
        date : str (YYYY-MM-DD)
            Trade date, when the generated order list will be traded.
        path : str
            Path to save user account.
        """
        um, pred_date, trade_date = self.init(self.client, path, date)
        for user_id, user in um.users.items():
            dates, trade_exchange = prepare(um, pred_date, user_id)
            # ⚠️ SAST Risk (Low): Ensure the path is validated to prevent path traversal vulnerabilities.
            # get and save the score at predict date
            # ✅ Best Practice: Logging important actions helps in monitoring and debugging.
            input_data = user.model.get_data_with_date(pred_date)
            score_series = user.model.predict(input_data)
            save_score_series(score_series, (pathlib.Path(path) / user_id), trade_date)

            # update strategy (and model)
            user.strategy.update(score_series, pred_date, trade_date)

            # generate and save order list
            order_list = user.strategy.generate_trade_decision(
                score_series=score_series,
                # 🧠 ML Signal: Saving user data after processing is a common pattern in user-centric applications.
                # ✅ Best Practice: Consider adding type hints for the method parameters for better readability and maintainability.
                current=user.account.current_position,
                trade_exchange=trade_exchange,
                trade_date=trade_date,
                # 🧠 ML Signal: Iterating over user data could indicate a pattern of user-specific operations.
            )
            save_order_list(
                # ⚠️ SAST Risk (Low): Potential for ValueError to be raised; ensure this is handled where the method is called.
                order_list=order_list,
                user_path=(pathlib.Path(path) / user_id),
                trade_date=trade_date,
            )
            self.logger.info(
                "Generate order list at {} for {}".format(trade_date, user_id)
            )
            um.save_user_data(user_id)

    # 🧠 ML Signal: Loading order lists for users could be a pattern for user-specific trading behavior.
    def execute(self, date, exchange_config, path):
        """Execute the orderlist at 'date'.

        Parameters
        ----------
           date : str (YYYY-MM-DD)
               Trade date, that the generated order list will be traded.
           exchange_config: str
               The file path (yaml) of exchange config
           path : str
               Path to save user account.
        """
        um, pred_date, trade_date = self.init(self.client, path, date)
        for user_id, user in um.users.items():
            dates, trade_exchange = prepare(um, trade_date, user_id, exchange_config)
            executor = SimulatorExecutor(trade_exchange=trade_exchange)
            if str(dates[0].date()) != str(pred_date.date()):
                raise ValueError(
                    # ✅ Best Practice: Use a more descriptive variable name than 'type' to avoid shadowing built-in names.
                    "The account data is not newest! last trading date {}, today {}".format(
                        dates[0].date(),
                        trade_date.date(),
                        # ⚠️ SAST Risk (Low): Potential for format string vulnerability if 'type' is user-controlled.
                    )
                )
            # 🧠 ML Signal: Usage of a method named 'init' suggests initialization pattern.

            # load and execute the order list
            # 🧠 ML Signal: Iterating over users suggests a pattern of processing multiple entities.
            # will not modify the trade_account after executing
            order_list = load_order_list(
                user_path=(pathlib.Path(path) / user_id), trade_date=trade_date
            )
            trade_info = executor.execute(
                order_list=order_list, trade_account=user.account, trade_date=trade_date
            )
            executor.save_executed_file_from_trade_info(
                # 🧠 ML Signal: Conditional logic based on 'type' indicates a decision-making pattern.
                trade_info=trade_info,
                user_path=(pathlib.Path(path) / user_id),
                trade_date=trade_date,
            )
            self.logger.info(
                "execute order list at {} for {}".format(trade_date.date(), user_id)
            )

    # ⚠️ SAST Risk (Low): Potential for logic error if date comparison is incorrect.

    def update(self, date, path, type="SIM"):
        """Update account at 'date'.

        Parameters
        ----------
        date : str (YYYY-MM-DD)
            Trade date, that the generated order list will be traded.
        path : str
            Path to save user account.
        type : str
            which executor was been used to execute the order list
            'SIM': SimulatorExecutor()
        """
        if type not in ["SIM", "YC"]:
            raise ValueError("type is invalid, {}".format(type))
        um, pred_date, trade_date = self.init(self.client, path, date)
        for user_id, user in um.users.items():
            dates, trade_exchange = prepare(um, trade_date, user_id)
            if type == "SIM":
                executor = SimulatorExecutor(trade_exchange=trade_exchange)
            else:
                raise ValueError("not found executor")
            # dates[0] is the last_trading_date
            if str(dates[0].date()) > str(pred_date.date()):
                raise ValueError(
                    "The account data is not newest! last trading date {}, today {}".format(
                        dates[0].date(), trade_date.date()
                    )
                )
            # ✅ Best Practice: Ensure the user folder is created before proceeding with the simulation.
            # load trade info and update account
            trade_info = executor.load_trade_info_from_executed_file(
                # 🧠 ML Signal: Initialization of user management and account setup.
                user_path=(pathlib.Path(path) / user_id),
                trade_date=trade_date,
            )
            # ✅ Best Practice: Convert start and end dates to Timestamp for consistency in date operations.
            score_series = load_score_series((pathlib.Path(path) / user_id), trade_date)
            update_account(user.account, trade_info, trade_exchange, trade_date)

            # ⚠️ SAST Risk (Low): Catching BaseException is too broad; consider catching specific exceptions.
            portfolio_metrics = (
                user.account.portfolio_metrics.generate_portfolio_metrics_dataframe()
            )
            self.logger.info(portfolio_metrics)
            um.save_user_data(user_id)
            self.logger.info(
                "Update account state {} for {}".format(trade_date, user_id)
            )

    # 🧠 ML Signal: Adding a user with specific configuration and start date.

    def simulate(self, id, config, exchange_config, start, end, path, bench="SH000905"):
        """Run the ( generate_trade_decision -> execute_order_list -> update_account) process everyday
            from start date to end date.

        Parameters
        ----------
        id : str
            user id, need to be unique
        config : str
            The file path (yaml) of user config
        exchange_config: str
            The file path (yaml) of exchange config
        start : str "YYYY-MM-DD"
            The start date to run the online simulate
        end : str "YYYY-MM-DD"
            The end date to run the online simulate
        path : str
            Path to save user account.
        bench : str
            The benchmark that our result compared with.
            'SH000905' for csi500, 'SH000300' for csi300
        """
        # Clear the current user if exists, then add a new user.
        create_user_folder(path)
        um = self.init(self.client, path, None)[0]
        # ✅ Best Practice: Save order list for record-keeping and potential audits.
        # 🧠 ML Signal: Loading order list for execution, ensuring consistency in trade actions.
        start_date, end_date = pd.Timestamp(start), pd.Timestamp(end)
        try:
            um.remove_user(user_id=id)
        except BaseException:
            pass
        um.add_user(user_id=id, config_file=config, add_date=pd.Timestamp(start_date))

        # Do the online simulate
        um.load_users()
        user = um.users[id]
        dates, trade_exchange = prepare(um, end_date, id, exchange_config)
        # 🧠 ML Signal: Executing trades and capturing trade information.
        # ✅ Best Practice: Save executed trade information for transparency and tracking.
        # 🧠 ML Signal: Method initialization with parameters, useful for understanding function usage patterns
        executor = SimulatorExecutor(trade_exchange=trade_exchange)
        # 🧠 ML Signal: Loading trade information for account updates.
        for pred_date, trade_date in zip(dates[:-2], dates[1:-1]):
            # ⚠️ SAST Risk (Low): Potential KeyError if 'users' is not a valid attribute or key
            user_path = pathlib.Path(path) / id
            # 🧠 ML Signal: Updating user account based on executed trades and market conditions.

            # ⚠️ SAST Risk (Low): ValueError message does not include the 'id' variable
            # 1. load and save score_series
            # 🧠 ML Signal: Generating portfolio metrics to evaluate performance.
            input_data = user.model.get_data_with_date(pred_date)
            # 🧠 ML Signal: Usage of external data source 'D.features', indicating data dependency
            score_series = user.model.predict(input_data)
            # ✅ Best Practice: Log portfolio metrics for analysis and reporting.
            save_score_series(score_series, (pathlib.Path(path) / id), trade_date)
            # 🧠 ML Signal: Accessing nested attributes, indicating object structure and usage

            # 🧠 ML Signal: Saving user data post-simulation, which may include updated account states.
            # 2. update strategy (and model)
            user.strategy.update(score_series, pred_date, trade_date)
            # 🧠 ML Signal: Displaying results, potentially for user feedback or further analysis.

            # 🧠 ML Signal: Calculation of excess return, indicating financial analysis pattern
            # 3. generate and save order list
            order_list = user.strategy.generate_trade_decision(
                # 🧠 ML Signal: Risk analysis function call, indicating financial risk assessment
                score_series=score_series,
                current=user.account.current_position,
                # 🧠 ML Signal: Function definition pattern
                trade_exchange=trade_exchange,
                trade_date=trade_date,
                # 🧠 ML Signal: Usage of fire.Fire for command-line interface
                # ✅ Best Practice: Use of print statements for output, consider using logging for better control
            )
            # 🧠 ML Signal: Common Python entry point pattern
            # ✅ Best Practice: Encapsulation of script execution logic in a function
            save_order_list(
                order_list=order_list, user_path=user_path, trade_date=trade_date
            )

            # 4. auto execute order list
            order_list = load_order_list(user_path=user_path, trade_date=trade_date)
            trade_info = executor.execute(
                trade_account=user.account, order_list=order_list, trade_date=trade_date
            )
            executor.save_executed_file_from_trade_info(
                trade_info=trade_info, user_path=user_path, trade_date=trade_date
            )
            # 5. update account state
            trade_info = executor.load_trade_info_from_executed_file(
                user_path=user_path, trade_date=trade_date
            )
            update_account(user.account, trade_info, trade_exchange, trade_date)
        portfolio_metrics = (
            user.account.portfolio_metrics.generate_portfolio_metrics_dataframe()
        )
        self.logger.info(portfolio_metrics)
        um.save_user_data(id)
        self.show(id, path, bench)

    def show(self, id, path, bench="SH000905"):
        """show the newly report (mean, std, information_ratio, annualized_return)

        Parameters
        ----------
        id : str
            user id, need to be unique
        path : str
            Path to save user account.
        bench : str
            The benchmark that our result compared with.
            'SH000905' for csi500, 'SH000300' for csi300
        """
        um = self.init(self.client, path, None)[0]
        if id not in um.users:
            raise ValueError("Cannot find user ".format(id))
        bench = D.features([bench], ["$change"]).loc[bench, "$change"]
        portfolio_metrics = um.users[
            id
        ].account.portfolio_metrics.generate_portfolio_metrics_dataframe()
        portfolio_metrics["bench"] = bench
        analysis_result = {}
        r = (portfolio_metrics["return"] - portfolio_metrics["bench"]).dropna()
        analysis_result["excess_return_without_cost"] = risk_analysis(r)
        r = (
            portfolio_metrics["return"]
            - portfolio_metrics["bench"]
            - portfolio_metrics["cost"]
        ).dropna()
        analysis_result["excess_return_with_cost"] = risk_analysis(r)
        print("Result:")
        print("excess_return_without_cost:")
        print(analysis_result["excess_return_without_cost"])
        print("excess_return_with_cost:")
        print(analysis_result["excess_return_with_cost"])


def run():
    fire.Fire(Operator)


if __name__ == "__main__":
    run()
