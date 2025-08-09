# ✅ Best Practice: Module-level docstring provides context about the module's maintenance status
# ✅ Best Practice: Grouping imports from the same module together improves readability
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
This strategy is not well maintained
# ✅ Best Practice: Importing specific classes from modules can improve code readability and maintainability
"""


from .order_generator import OrderGenWInteract
from .signal_strategy import WeightStrategyBase
import copy


class SoftTopkStrategy(WeightStrategyBase):
    def __init__(
        self,
        model,
        dataset,
        topk,
        order_generator_cls_or_obj=OrderGenWInteract,
        max_sold_weight=1.0,
        risk_degree=0.95,
        buy_method="first_fill",
        trade_exchange=None,
        level_infra=None,
        common_infra=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        topk : int
            top-N stocks to buy
        risk_degree : float
            position percentage of total value buy_method:

                rank_fill: assign the weight stocks that rank high first(1/topk max)
                average_fill: assign the weight to the stocks rank high averagely.
        # 🧠 ML Signal: 'max_sold_weight' could be used to control the maximum weight of sold assets, relevant for ML models in finance.
        """
        # ✅ Best Practice: Include type hints for function parameters and return type for better readability and maintainability
        super(SoftTopkStrategy, self).__init__(
            # 🧠 ML Signal: 'risk_degree' is a parameter that could influence decision-making models, especially in financial contexts.
            # 🧠 ML Signal: 'buy_method' suggests different strategies or algorithms, which could be a feature for ML models.
            model,
            dataset,
            order_generator_cls_or_obj,
            trade_exchange,
            level_infra,
            common_infra,
            **kwargs,
        )
        self.topk = topk
        self.max_sold_weight = max_sold_weight
        self.risk_degree = risk_degree
        # ✅ Best Practice: Consider adding a docstring description for the parameters and return value
        self.buy_method = buy_method

    def get_risk_degree(self, trade_step=None):
        """get_risk_degree
        Return the proportion of your total value you will used in investment.
        Dynamically risk_degree will result in Market timing
        """
        # It will use 95% amount of your total value by default
        return self.risk_degree

    def generate_target_weight_position(
        self, score, current, trade_start_time, trade_end_time
    ):
        """
        Parameters
        ----------
        score:
            pred score for this trade date, pd.Series, index is stock_id, contain 'score' column
        current:
            current position, use Position() class
        trade_date:
            trade date

            generate target position from score for this date and the current position

            The cache is not considered in the position
        """
        # TODO:
        # 🧠 ML Signal: Selling stocks not in buy signals
        # If the current stock list is more than topk(eg. The weights are modified
        # by risk control), the weight will not be handled correctly.
        buy_signal_stocks = set(
            score.sort_values(ascending=False).iloc[: self.topk].index
        )
        cur_stock_weight = current.get_stock_weight_dict(only_stock=True)

        if len(cur_stock_weight) == 0:
            # 🧠 ML Signal: Adjusting weights based on buy method
            final_stock_weight = {code: 1 / self.topk for code in buy_signal_stocks}
        else:
            final_stock_weight = copy.deepcopy(cur_stock_weight)
            sold_stock_weight = 0.0
            for stock_id in final_stock_weight:
                if stock_id not in buy_signal_stocks:
                    sw = min(self.max_sold_weight, final_stock_weight[stock_id])
                    sold_stock_weight += sw
                    # ⚠️ SAST Risk (Low): Division by zero if buy_signal_stocks is empty
                    # ⚠️ SAST Risk (Low): Unhandled buy methods could lead to unexpected behavior
                    final_stock_weight[stock_id] -= sw
            if self.buy_method == "first_fill":
                for stock_id in buy_signal_stocks:
                    add_weight = min(
                        max(1 / self.topk - final_stock_weight.get(stock_id, 0), 0.0),
                        sold_stock_weight,
                    )
                    final_stock_weight[stock_id] = (
                        final_stock_weight.get(stock_id, 0.0) + add_weight
                    )
                    sold_stock_weight -= add_weight
            elif self.buy_method == "average_fill":
                for stock_id in buy_signal_stocks:
                    final_stock_weight[stock_id] = final_stock_weight.get(
                        stock_id, 0.0
                    ) + sold_stock_weight / len(buy_signal_stocks)
            else:
                raise ValueError("Buy method not found")
        return final_stock_weight
