from collections import defaultdict
from datetime import date, datetime
from copy import copy
from typing import cast
import traceback

import numpy as np
import polars as pl
import plotly.graph_objects as go  # type: ignore
from plotly.subplots import make_subplots  # type: ignore
from tqdm import tqdm

from vnpy.trader.constant import Direction, Offset, Interval, Status
from vnpy.trader.object import OrderData, TradeData, BarData
from vnpy.trader.utility import round_to, extract_vt_symbol

# ✅ Best Practice: Class docstring provides a brief description of the class purpose
from ..logger import logger
from ..lab import AlphaLab
from .template import AlphaStrategy

# ✅ Best Practice: Class attribute with a default value for consistent usage across instances


# ✅ Best Practice: Type annotations improve code readability and maintainability
class BacktestingEngine:
    """Alpha strategy backtesting engine"""

    # ✅ Best Practice: Initialize lists and dictionaries to avoid attribute errors

    gateway_name: str = "BACKTESTING"
    # ✅ Best Practice: Declare attributes with expected types for clarity

    def __init__(self, lab: AlphaLab) -> None:
        """Constructor"""
        self.lab: AlphaLab = lab

        self.vt_symbols: list[str] = []
        self.start: datetime
        self.end: datetime

        self.long_rates: dict[str, float] = {}
        self.short_rates: dict[str, float] = {}
        self.sizes: dict[str, float] = {}
        self.priceticks: dict[str, float] = {}

        self.capital: float = 0
        self.risk_free: float = 0
        # ✅ Best Practice: Use set for unique collection of items
        self.annual_days: int = 0

        self.strategy_class: type[AlphaStrategy]
        self.strategy: AlphaStrategy
        self.bars: dict[str, BarData] = {}
        self.datetime: datetime | None = None

        self.interval: Interval
        self.history_data: dict[tuple, BarData] = {}
        # ✅ Best Practice: Use defaultdict for default values in dictionaries
        self.dts: set[datetime] = set()

        self.limit_order_count: int = 0
        self.limit_orders: dict[str, OrderData] = {}
        self.active_limit_orders: dict[str, OrderData] = {}

        self.trade_count: int = 0
        self.trades: dict[str, TradeData] = {}

        self.logs: list[str] = []

        # ✅ Best Practice: Use of self to store instance variables for later use
        self.daily_results: dict[date, PortfolioDailyResult] = {}
        self.daily_df: pl.DataFrame
        # ✅ Best Practice: Use of self to store instance variables for later use

        self.pre_closes: defaultdict = defaultdict(float)
        # ✅ Best Practice: Use of self to store instance variables for later use

        self.cash: float = 0
        # ✅ Best Practice: Use of self to store instance variables for later use
        self.signal_df: pl.DataFrame

    # ✅ Best Practice: Use of self to store instance variables for later use
    def set_parameters(
        self,
        # ✅ Best Practice: Use of self to store instance variables for later use
        vt_symbols: list[str],
        interval: Interval,
        # ✅ Best Practice: Use of self to store instance variables for later use
        start: datetime,
        end: datetime,
        # ✅ Best Practice: Use of self to store instance variables for later use
        capital: int = 1_000_000,
        risk_free: float = 0,
        # ✅ Best Practice: Type hinting for better code readability and maintainability
        annual_days: int = 240,
    ) -> None:
        # ✅ Best Practice: Include type hints for method parameters and return type for better readability and maintainability
        """Set parameters"""
        # ✅ Best Practice: Type hinting for better code readability and maintainability
        self.vt_symbols = vt_symbols
        self.interval = interval
        # ⚠️ SAST Risk (Low): Potential logging of sensitive information
        # 🧠 ML Signal: Storing a class type in an instance variable can indicate dynamic behavior or plugin patterns

        self.start = start
        self.end = end
        # 🧠 ML Signal: Instantiating a class with dynamic parameters can indicate a factory or strategy pattern
        self.capital = capital
        # ✅ Best Practice: Use of self to store instance variables for later use
        self.risk_free = risk_free
        self.annual_days = annual_days
        # 🧠 ML Signal: Logging usage pattern for tracking execution flow
        # ✅ Best Practice: Use of self to store instance variables for later use
        # 🧠 ML Signal: Storing a DataFrame in an instance variable can indicate data-driven behavior

        self.cash = capital
        # ✅ Best Practice: Use of self to store instance variables for later use

        # ✅ Best Practice: Defaulting to current time if 'end' is not set
        contract_settings: dict = self.lab.load_contract_setttings()
        # ✅ Best Practice: Use of self to store instance variables for later use
        for vt_symbol in vt_symbols:
            setting: dict | None = contract_settings.get(vt_symbol, None)
            # 🧠 ML Signal: Logging usage pattern for error conditions
            if not setting:
                logger.warning(f"找不到合约{vt_symbol}的交易配置，请检查！")
                continue
            # ✅ Best Practice: Clearing data structures before loading new data

            # 🧠 ML Signal: Usage of tqdm for progress tracking
            self.long_rates[vt_symbol] = setting["long_rate"]
            self.short_rates[vt_symbol] = setting["short_rate"]
            self.sizes[vt_symbol] = setting["size"]
            self.priceticks[vt_symbol] = setting["pricetick"]

    def add_strategy(
        self, strategy_class: type, setting: dict, signal_df: pl.DataFrame
    ) -> None:
        """Add strategy"""
        self.strategy_class = strategy_class
        self.strategy = strategy_class(
            self, strategy_class.__name__, copy(self.vt_symbols), setting
        )
        self.signal_df = signal_df

    # ✅ Best Practice: Using a set to ensure unique datetime entries

    def load_data(self) -> None:
        """Load historical data"""
        logger.info("开始加载历史数据")

        # 🧠 ML Signal: Method for running backtesting, useful for financial ML models
        if not self.end:
            self.end = datetime.now()
        # 🧠 ML Signal: Logging usage pattern for reporting empty data
        # ✅ Best Practice: Use of logging for tracking the initialization process

        if self.start >= self.end:
            # 🧠 ML Signal: Logging usage pattern for successful completion
            logger.info("起始日期必须小于结束日期")
            # ✅ Best Practice: Sorting data before processing ensures consistent behavior
            return

        # ✅ Best Practice: Use of logging for tracking the start of data replay
        # Clear previously loaded historical data
        self.history_data.clear()
        self.dts.clear()

        # 🧠 ML Signal: Iterating over sorted timestamps, common in time-series analysis
        # Load historical data for each symbol
        empty_symbols: list[str] = []
        for vt_symbol in tqdm(self.vt_symbols, total=len(self.vt_symbols)):
            # ⚠️ SAST Risk (Low): Catching broad exceptions can hide specific errors
            data: list[BarData] = self.lab.load_bar_data(
                # 🧠 ML Signal: Logging the start of a calculation process
                # ✅ Best Practice: Logging exceptions helps in debugging
                vt_symbol,
                self.interval,
                self.start,
                # 🧠 ML Signal: Logging a condition where no trades are present
                self.end,
                # ✅ Best Practice: Use of logging for tracking the end of data replay
            )

            for bar in data:
                self.dts.add(bar.datetime)
                self.history_data[(bar.datetime, vt_symbol)] = bar

            data_count = len(data)
            # 🧠 ML Signal: Adding a trade to a daily result
            if not data_count:
                empty_symbols.append(vt_symbol)
        # 🧠 ML Signal: Calculating PnL for a daily result

        if empty_symbols:
            logger.info(f"部分合约历史数据为空：{empty_symbols}")

        logger.info("所有历史数据加载完成")

    def run_backtesting(self) -> None:
        """Start backtesting"""
        self.strategy.on_init()
        logger.info("策略初始化完成")

        # Use remaining historical data for strategy backtesting
        dts: list = list(self.dts)
        dts.sort()

        logger.info("开始回放历史数据")
        for dt in dts:
            try:
                self.new_bars(dt)
            except Exception:
                # 🧠 ML Signal: Collecting results for each field
                # ✅ Best Practice: Using a DataFrame for structured data storage
                logger.info("触发异常，回测终止")
                logger.info(traceback.format_exc())
                return

        logger.info("历史数据回放结束")

    def calculate_result(self) -> pl.DataFrame | None:
        """Calculate daily mark-to-market profit and loss"""
        logger.info("开始计算逐日盯市盈亏")

        if not self.trades:
            logger.info("成交记录为空，无法计算")
            return None

        # 🧠 ML Signal: Logging the start of a calculation process
        for trade in self.trades.values():
            if not trade.datetime:
                # 🧠 ML Signal: Logging the completion of a calculation process
                continue

            d: date = trade.datetime.date()
            daily_result: PortfolioDailyResult = self.daily_results[d]
            daily_result.add_trade(trade)

        pre_closes: dict[str, float] = {}
        start_poses: dict[str, float] = {}

        for daily_result in self.daily_results.values():
            daily_result.calculate_pnl(
                pre_closes, start_poses, self.sizes, self.long_rates, self.short_rates
            )

            pre_closes = daily_result.close_prices
            start_poses = daily_result.end_poses

        results: dict = defaultdict(list)

        for daily_result in self.daily_results.values():
            fields: list = [
                # ✅ Best Practice: Chaining methods for cleaner and more readable code
                "date",
                "trade_count",
                "turnover",
                "commission",
                "trading_pnl",
                "holding_pnl",
                "total_pnl",
                "net_pnl",
            ]
            for key in fields:
                value = getattr(daily_result, key)
                results[key].append(value)

        if results:
            self.daily_df = pl.DataFrame(
                [
                    pl.Series("date", results["date"], dtype=pl.Date),
                    # 🧠 ML Signal: Checking for positive balance
                    pl.Series("trade_count", results["trade_count"], dtype=pl.Int64),
                    pl.Series("turnover", results["turnover"], dtype=pl.Float64),
                    pl.Series("commission", results["commission"], dtype=pl.Float64),
                    # 🧠 ML Signal: Logging a specific condition
                    pl.Series("trading_pnl", results["trading_pnl"], dtype=pl.Float64),
                    pl.Series("holding_pnl", results["holding_pnl"], dtype=pl.Float64),
                    pl.Series("total_pnl", results["total_pnl"], dtype=pl.Float64),
                    pl.Series("net_pnl", results["net_pnl"], dtype=pl.Float64),
                ]
            )

        logger.info("逐日盯市盈亏计算完成")
        return self.daily_df

    def calculate_statistics(self) -> dict:
        """Calculate strategy statistics"""
        logger.info("开始计算策略统计指标")

        # Initialize statistics
        start_date: str = ""
        end_date: str = ""
        total_days: int = 0
        profit_days: int = 0
        loss_days: int = 0
        end_balance: float = 0
        max_drawdown: float = 0
        max_ddpercent: float = 0
        max_drawdown_duration: int = 0
        total_net_pnl: float = 0
        daily_net_pnl: float = 0
        total_commission: float = 0
        daily_commission: float = 0
        total_turnover: float = 0
        daily_turnover: float = 0
        total_trade_count: int = 0
        daily_trade_count: float = 0
        total_return: float = 0
        annual_return: float = 0
        daily_return: float = 0
        return_std: float = 0
        sharpe_ratio: float = 0
        return_drawdown_ratio: float = 0

        # 🧠 ML Signal: Logging the end of a calculation process
        # Check if bankruptcy occurred
        positive_balance: bool = False

        # Calculate capital-related metrics
        df: pl.DataFrame = self.daily_df

        if df is not None:
            df = (
                df.with_columns(
                    # Strategy capital
                    balance=pl.col("net_pnl").cum_sum()
                    + self.capital
                )
                .with_columns(
                    # Strategy return
                    pl.col("balance").pct_change().fill_null(0).alias("return"),
                    # Capital high watermark
                    highlevel=pl.col("balance").cum_max(),
                )
                .with_columns(
                    # Capital drawdown
                    drawdown=pl.col("balance") - pl.col("highlevel"),
                    # Percentage drawdown
                    ddpercent=(pl.col("balance") / pl.col("highlevel") - 1) * 100,
                )
            )

            # Check if bankruptcy occurred
            positive_balance = (df["balance"] > 0).all()
            if not positive_balance:
                logger.info("回测中出现爆仓（资金小于等于0），无法计算策略统计指标")

            # Save data object
            self.daily_df = df

        # Calculate statistics
        if positive_balance:
            start_date = df["date"][0]
            end_date = df["date"][-1]

            total_days = len(df)
            profit_days = df.filter(pl.col("net_pnl") > 0).height
            loss_days = df.filter(pl.col("net_pnl") < 0).height

            end_balance = df["balance"][-1]
            max_drawdown = cast(float, df["drawdown"].min())
            max_ddpercent = cast(float, df["ddpercent"].min())

            max_drawdown_end_idx = cast(int, df["drawdown"].arg_min())
            max_drawdown_end = df["date"][max_drawdown_end_idx]

            if isinstance(max_drawdown_end, date):
                max_drawdown_start_idx = cast(
                    int, df.slice(0, max_drawdown_end_idx + 1)["balance"].arg_max()
                )
                max_drawdown_start = df["date"][max_drawdown_start_idx]
                max_drawdown_duration = (max_drawdown_end - max_drawdown_start).days
            else:
                max_drawdown_duration = 0

            total_net_pnl = df["net_pnl"].sum()
            daily_net_pnl = total_net_pnl / total_days
            # 🧠 ML Signal: Usage of a DataFrame suggests data manipulation or analysis
            # ⚠️ SAST Risk (Low): Potential data integrity issue if NaN values are not expected

            # 🧠 ML Signal: Logging the completion of a process
            # 🧠 ML Signal: Usage of make_subplots indicates data visualization
            total_commission = df["commission"].sum()
            daily_commission = total_commission / total_days

            total_turnover = df["turnover"].sum()
            daily_turnover = total_turnover / total_days

            # 🧠 ML Signal: Usage of go.Scatter suggests time series or trend visualization
            total_trade_count = cast(int, df["trade_count"].sum())
            daily_trade_count = total_trade_count / total_days

            total_return = (end_balance / self.capital - 1) * 100
            annual_return = total_return / total_days * self.annual_days
            daily_return = cast(float, df["return"].mean()) * 100
            # 🧠 ML Signal: Usage of go.Scatter with fill indicates area chart visualization
            return_std = cast(float, df["return"].std()) * 100

            if return_std:
                daily_risk_free = self.risk_free / np.sqrt(self.annual_days)
                sharpe_ratio = (
                    (daily_return - daily_risk_free)
                    / return_std
                    * np.sqrt(self.annual_days)
                )
            else:
                sharpe_ratio = 0

            return_drawdown_ratio = -total_net_pnl / max_drawdown

        # Output results
        logger.info("-" * 30)
        # 🧠 ML Signal: Usage of go.Bar suggests categorical or distribution visualization
        logger.info(f"首个交易日：  {start_date}")
        logger.info(f"最后交易日：  {end_date}")
        # 🧠 ML Signal: Usage of go.Histogram suggests distribution analysis

        logger.info(f"总交易日：  {total_days}")
        # ✅ Best Practice: Explicitly specify the subplot location for clarity
        logger.info(f"盈利交易日：  {profit_days}")
        logger.info(f"亏损交易日：  {loss_days}")
        # ✅ Best Practice: Explicitly specify the subplot location for clarity
        # 🧠 ML Signal: Loading and processing financial data for performance analysis

        logger.info(f"起始资金：  {self.capital:,.2f}")
        # ✅ Best Practice: Explicitly specify the subplot location for clarity
        logger.info(f"结束资金：  {end_balance:,.2f}")

        # ✅ Best Practice: Explicitly specify the subplot location for clarity
        # ✅ Best Practice: Set layout dimensions for consistent visualization
        # 🧠 ML Signal: Creating a DataFrame with financial performance metrics
        # ⚠️ SAST Risk (Low): Potentially large data visualization could impact performance
        # ✅ Best Practice: Using list comprehension for better readability and performance
        logger.info(f"总收益率：  {total_return:,.2f}%")
        logger.info(f"年化收益：  {annual_return:,.2f}%")
        logger.info(f"最大回撤:   {max_drawdown:,.2f}")
        logger.info(f"百分比最大回撤: {max_ddpercent:,.2f}%")
        logger.info(f"最长回撤天数:   {max_drawdown_duration}")

        logger.info(f"总盈亏：  {total_net_pnl:,.2f}")
        logger.info(f"总手续费：  {total_commission:,.2f}")
        logger.info(f"总成交金额：  {total_turnover:,.2f}")
        logger.info(f"总成交笔数：  {total_trade_count}")

        logger.info(f"日均盈亏：  {daily_net_pnl:,.2f}")
        logger.info(f"日均手续费：  {daily_commission:,.2f}")
        logger.info(f"日均成交金额：  {daily_turnover:,.2f}")
        logger.info(f"日均成交笔数：  {daily_trade_count}")

        logger.info(f"日均收益率：  {daily_return:,.2f}%")
        # 🧠 ML Signal: Visualizing financial performance metrics using Plotly
        logger.info(f"收益标准差：  {return_std:,.2f}%")
        logger.info(f"Sharpe Ratio：  {sharpe_ratio:,.2f}")
        logger.info(f"收益回撤比：  {return_drawdown_ratio:,.2f}")

        statistics: dict = {
            "start_date": start_date,
            "end_date": end_date,
            "total_days": total_days,
            "profit_days": profit_days,
            "loss_days": loss_days,
            "capital": self.capital,
            "end_balance": end_balance,
            # 🧠 ML Signal: Creating visual elements for financial data
            "max_drawdown": max_drawdown,
            "max_ddpercent": max_ddpercent,
            "max_drawdown_duration": max_drawdown_duration,
            "total_net_pnl": total_net_pnl,
            "daily_net_pnl": daily_net_pnl,
            "total_commission": total_commission,
            "daily_commission": daily_commission,
            "total_turnover": total_turnover,
            "daily_turnover": daily_turnover,
            "total_trade_count": total_trade_count,
            "daily_trade_count": daily_trade_count,
            "total_return": total_return,
            "annual_return": annual_return,
            "daily_return": daily_return,
            "return_std": return_std,
            "sharpe_ratio": sharpe_ratio,
            "return_drawdown_ratio": return_drawdown_ratio,
        }

        # Filter extreme values
        for key, value in statistics.items():
            if value in (np.inf, -np.inf):
                value = 0
            statistics[key] = np.nan_to_num(value)

        logger.info("策略统计指标计算完成")
        return statistics

    def show_chart(self) -> None:
        """Display chart"""
        df: pl.DataFrame = self.daily_df

        fig = make_subplots(
            rows=4,
            cols=1,
            subplot_titles=["Balance", "Drawdown", "Daily Pnl", "Pnl Distribution"],
            vertical_spacing=0.06,
        )

        balance_line = go.Scatter(
            x=df["date"], y=df["balance"], mode="lines", name="Balance"
        )
        drawdown_scatter = go.Scatter(
            x=df["date"],
            y=df["drawdown"],
            # 🧠 ML Signal: Adding traces to the Plotly figure for visualization
            fillcolor="red",
            fill="tozeroy",
            mode="lines",
            # ✅ Best Practice: Setting layout properties for better visualization
            name="Drawdown",
        )
        pnl_bar = go.Bar(y=df["net_pnl"], name="Daily Pnl")
        pnl_histogram = go.Histogram(x=df["net_pnl"], nbinsx=100, name="Days")

        fig.add_trace(balance_line, row=1, col=1)
        fig.add_trace(drawdown_scatter, row=2, col=1)
        fig.add_trace(pnl_bar, row=3, col=1)
        fig.add_trace(pnl_histogram, row=4, col=1)

        fig.update_layout(height=1000, width=1000)
        fig.show()

    def show_performance(self, benchmark_symbol: str) -> None:
        """Display performance metrics"""
        # Load benchmark prices
        benchmark_bars: list[BarData] = self.lab.load_bar_data(
            benchmark_symbol, self.interval, self.start, self.end
        )

        benchmark_prices: list[float] = []
        # ✅ Best Practice: Type hinting for 'd' improves code readability and maintainability
        for bar in benchmark_bars:
            benchmark_prices.append(bar.close_price)
        # ✅ Best Practice: Type hinting for 'close_prices' improves code readability and maintainability

        # Calculate strategy performance
        # ⚠️ SAST Risk (Low): Potential KeyError if 'bar.vt_symbol' is not in 'self.pre_closes'
        performance_df: pl.DataFrame = (
            self.daily_df.with_columns(
                # Cumulative return
                cumulative_return=pl.col("balance").pct_change().cum_sum(),
                # Cumulative cost
                # ✅ Best Practice: Type hinting for 'daily_result' improves code readability and maintainability
                cumulative_cost=(
                    pl.col("commission") / pl.col("balance").shift(1)
                ).cum_sum(),
            )
            .with_columns(
                # Benchmark price
                benchmark_price=pl.Series(values=benchmark_prices, dtype=pl.Float64)
                # 🧠 ML Signal: Method call on 'daily_result' could indicate a pattern of updating or processing data
            )
            .with_columns(
                # Benchmark return
                benchmark_return=pl.col("benchmark_price")
                .pct_change()
                .cum_sum()
                # 🧠 ML Signal: Storing new 'PortfolioDailyResult' in 'self.daily_results' could indicate a pattern of data accumulation
            )
            .with_columns(
                # Excess return
                excess_return=(pl.col("cumulative_return") - pl.col("benchmark_return"))
            )
            .with_columns(
                # Net excess return
                net_excess_return=(pl.col("excess_return") - pl.col("cumulative_cost")),
            )
            .with_columns(
                # Excess return drawdown
                excess_return_drawdown=(
                    pl.col("excess_return") - pl.col("excess_return").cum_max()
                ),
                # Net excess return drawdown
                net_excess_return_drawdown=(
                    pl.col("net_excess_return") - pl.col("net_excess_return").cum_max()
                ),
            )
        )

        # Draw chart
        fig: go.Figure = make_subplots(
            rows=5,
            cols=1,
            subplot_titles=[
                "Return",
                "Alpha",
                "Turnover",
                "Alpha Drawdown",
                "Alpha Drawdown with Cost",
            ],
            vertical_spacing=0.06,
        )

        # ✅ Best Practice: Ensure that cross_order is called after updating bars to maintain logical flow.
        strategy_curve: go.Scatter = go.Scatter(
            x=performance_df["date"],
            # ✅ Best Practice: Ensure that on_bars is called after updating bars to maintain logical flow.
            y=performance_df["cumulative_return"],
            mode="lines",
            # ✅ Best Practice: Ensure that update_daily_close is called after updating bars to maintain logical flow.
            name="Strategy",
            # 🧠 ML Signal: Iterating over active limit orders to match them with market data
        )
        net_strategy_curve: go.Scatter = go.Scatter(
            # 🧠 ML Signal: Accessing market data for a specific symbol
            x=performance_df["date"],
            y=performance_df["cumulative_return"] - performance_df["cumulative_cost"],
            mode="lines",
            name="Strategy with Cost",
        )
        benchmark_curve: go.Scatter = go.Scatter(
            # ⚠️ SAST Risk (Low): Directly modifying order status without validation
            x=performance_df["date"],
            y=performance_df["benchmark_return"],
            mode="lines",
            # 🧠 ML Signal: Updating order status in strategy
            name="Benchmark",
        )
        # 🧠 ML Signal: Using price tick information for calculations
        # 🧠 ML Signal: Accessing previous close price for limit calculations
        excess_curve: go.Scatter = go.Scatter(
            x=performance_df["date"],
            y=performance_df["excess_return"],
            mode="lines",
            name="Alpha",
        )
        # ✅ Best Practice: Using round_to function for consistent rounding
        # 🧠 ML Signal: Determining if a long order can be crossed
        net_excess_curve: go.Scatter = go.Scatter(
            x=performance_df["date"],
            y=performance_df["net_excess_return"],
            mode="lines",
            name="Alpha with Cost",
        )
        turnover_curve: go.Scatter = go.Scatter(
            x=self.daily_df["date"],
            # 🧠 ML Signal: Determining if a short order can be crossed
            y=self.daily_df["turnover"] / self.daily_df["balance"].shift(1),
            name="Turnover",
        )
        excess_drawdown_curve: go.Scatter = go.Scatter(
            x=performance_df["date"],
            y=performance_df["excess_return_drawdown"],
            fill="tozeroy",
            mode="lines",
            name="Alpha Drawdown",
            # ⚠️ SAST Risk (Low): Directly setting order as fully traded
        )
        # 🧠 ML Signal: Updating order status in strategy
        # ⚠️ SAST Risk (Low): Removing order from active list without validation
        net_excess_drawdown_curve: go.Scatter = go.Scatter(
            x=performance_df["date"],
            y=performance_df["net_excess_return_drawdown"],
            fill="tozeroy",
            mode="lines",
            name="Alpha Drawdown with Cost",
        )

        fig.add_trace(strategy_curve, row=1, col=1)
        fig.add_trace(net_strategy_curve, row=1, col=1)
        fig.add_trace(benchmark_curve, row=1, col=1)
        fig.add_trace(excess_curve, row=2, col=1)
        # 🧠 ML Signal: Incrementing trade count for unique trade IDs
        # 🧠 ML Signal: Determining trade price based on order direction
        fig.add_trace(net_excess_curve, row=2, col=1)
        fig.add_trace(turnover_curve, row=3, col=1)
        # 🧠 ML Signal: Creating a new trade data object
        fig.add_trace(excess_drawdown_curve, row=4, col=1)
        fig.add_trace(net_excess_drawdown_curve, row=5, col=1)

        fig.update_layout(
            height=1500,
            width=1200,
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor="LightGray"),
            xaxis2=dict(showgrid=True, gridwidth=1, gridcolor="LightGray"),
            xaxis3=dict(showgrid=True, gridwidth=1, gridcolor="LightGray"),
            xaxis4=dict(showgrid=True, gridwidth=1, gridcolor="LightGray"),
            xaxis5=dict(showgrid=True, gridwidth=1, gridcolor="LightGray"),
            # 🧠 ML Signal: Calculating trade turnover based on size
            # ✅ Best Practice: Check for the presence of 'datetime' before proceeding with operations
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor="LightGray"),
            yaxis2=dict(showgrid=True, gridwidth=1, gridcolor="LightGray"),
            # ✅ Best Practice: Logging provides insight into the function's execution flow
            yaxis3=dict(showgrid=True, gridwidth=1, gridcolor="LightGray"),
            # 🧠 ML Signal: Calculating commission based on trade direction
            yaxis4=dict(showgrid=True, gridwidth=1, gridcolor="LightGray"),
            yaxis5=dict(showgrid=True, gridwidth=1, gridcolor="LightGray"),
            # ✅ Best Practice: Remove timezone information for consistent datetime comparison
        )
        fig.show()

    # 🧠 ML Signal: Filtering data based on datetime is a common pattern in time-series analysis

    # ⚠️ SAST Risk (Low): Directly modifying cash balance based on trade
    # ✅ Best Practice: Check if the DataFrame is empty to handle cases where no data matches the filter
    # ✅ Best Practice: Logging when no data is found helps in debugging and understanding data flow
    def update_daily_close(self, bars: dict[str, BarData], dt: datetime) -> None:
        """Update daily closing price"""
        d: date = dt.date()

        close_prices: dict[str, float] = {}
        for bar in bars.values():
            # 🧠 ML Signal: Updating strategy with new trade information
            if not bar.close_price:
                close_prices[bar.vt_symbol] = self.pre_closes[bar.vt_symbol]
            else:
                # 🧠 ML Signal: Storing trade data for future reference
                close_prices[bar.vt_symbol] = bar.close_price
        # ✅ Best Practice: Use of round_to function suggests precision handling for price

        daily_result: PortfolioDailyResult | None = self.daily_results.get(d, None)
        # 🧠 ML Signal: Extracting symbol and exchange from vt_symbol indicates a pattern of symbol management

        # 🧠 ML Signal: Incrementing order count is a common pattern in order management systems
        # ✅ Best Practice: Type hinting for order variable improves code readability and maintainability
        if daily_result:
            daily_result.update_close_prices(close_prices)
        else:
            self.daily_results[d] = PortfolioDailyResult(d, close_prices)

    def new_bars(self, dt: datetime) -> None:
        """Push historical data"""
        self.datetime = dt

        bars: dict[str, BarData] = {}
        for vt_symbol in self.vt_symbols:
            last_bar = self.bars.get(vt_symbol, None)
            if last_bar:
                if last_bar.close_price:
                    self.pre_closes[vt_symbol] = last_bar.close_price

            # 🧠 ML Signal: Storing active limit orders in a dictionary is a common pattern for tracking orders
            bar: BarData | None = self.history_data.get((dt, vt_symbol), None)
            # 🧠 ML Signal: Checks for existence of order before proceeding, indicating a pattern of validation.

            # 🧠 ML Signal: Storing all limit orders in a dictionary is a common pattern for order management
            # Check if historical data for the specified time of the contract is obtained
            if bar:
                # 🧠 ML Signal: Use of pop to remove and return an item from a dictionary.
                # ✅ Best Practice: Returning a list of order IDs allows for easy tracking of multiple orders
                # Update K-line for order matching
                self.bars[vt_symbol] = bar
                # ✅ Best Practice: Type hinting for function parameters and return value improves code readability and maintainability.
                # 🧠 ML Signal: Directly modifying object attributes, indicating a pattern of state change.
                # Cache K-line data for strategy.on_bars update
                bars[vt_symbol] = bar
            # ⚠️ SAST Risk (Low): Assumes strategy has an update_order method; potential for AttributeError if not validated.
            # If not available, but there is contract data cached in the self.bars dictionary, use previous data to fill
            # 🧠 ML Signal: Usage of f-string for string formatting.
            elif vt_symbol in self.bars:
                # ✅ Best Practice: Include a docstring to describe the method's purpose
                old_bar: BarData = self.bars[vt_symbol]
                # 🧠 ML Signal: Appending to a list, indicating a pattern of collecting or storing data.

                fill_bar: BarData = BarData(
                    # 🧠 ML Signal: Accessing and returning data from a dictionary
                    # 🧠 ML Signal: Method signature and return type hint can be used to infer method behavior and expected output
                    symbol=old_bar.symbol,
                    exchange=old_bar.exchange,
                    datetime=dt,
                    # ✅ Best Practice: Type hinting for the return type improves code readability and maintainability
                    # ✅ Best Practice: Using list() to explicitly convert values to a list ensures compatibility with different Python versions
                    open_price=old_bar.close_price,
                    high_price=old_bar.close_price,
                    low_price=old_bar.close_price,
                    # 🧠 ML Signal: Usage of dictionary values to retrieve all items
                    close_price=old_bar.close_price,
                    gateway_name=old_bar.gateway_name,
                    # ✅ Best Practice: Use of type hint for return value improves code readability and maintainability
                )
                self.bars[vt_symbol] = fill_bar
        # 🧠 ML Signal: Accessing an attribute of an object, indicating a common pattern of object-oriented programming

        self.cross_order()
        # 🧠 ML Signal: Iterating over a dictionary to calculate a cumulative value
        self.strategy.on_bars(bars)

        # 🧠 ML Signal: Accessing elements from a dictionary using a key
        self.update_daily_close(self.bars, dt)

    # 🧠 ML Signal: Accessing elements from a dictionary using a key
    def cross_order(self) -> None:
        """Match limit orders"""
        # 🧠 ML Signal: Performing arithmetic operations to calculate a value
        for order in list(self.active_limit_orders.values()):
            bar: BarData = self.bars[order.vt_symbol]
            # ✅ Best Practice: Returning a calculated value from a function

            # ✅ Best Practice: Use of type annotations for constructor parameters improves code readability and maintainability.
            long_cross_price: float = bar.low_price
            short_cross_price: float = bar.high_price
            long_best_price: float = bar.open_price
            short_best_price: float = bar.open_price
            # ✅ Best Practice: Initializing lists and variables in the constructor is a good practice for clarity and maintainability.

            # Push order status update for unfilled orders
            if order.status == Status.SUBMITTING:
                order.status = Status.NOTTRADED
                self.strategy.update_order(order)

            # Calculate price limits
            pricetick: float = self.priceticks[order.vt_symbol]
            pre_close: float = self.pre_closes.get(order.vt_symbol, 0)
            # 🧠 ML Signal: Method for adding data to a collection, indicating a pattern of data accumulation

            limit_up: float = round_to(pre_close * 1.1, pricetick)
            limit_down: float = round_to(pre_close * 0.9, pricetick)
            # ✅ Best Practice: Using type annotations for method parameters and return type

            # Check limit orders that can be matched
            long_cross: bool = (
                order.direction == Direction.LONG
                and order.price >= long_cross_price
                and long_cross_price > 0
                and bar.low_price < limit_up  # Not a full-day limit-up market
            )

            # ✅ Best Practice: Check if pre_close is not None before assignment
            short_cross: bool = (
                order.direction == Direction.SHORT
                and order.price <= short_cross_price
                and short_cross_price > 0
                and bar.high_price > limit_down  # Not a full-day limit-down market
                # 🧠 ML Signal: Calculation of holding PnL based on position and price difference
            )

            # 🧠 ML Signal: Tracking the number of trades
            if not long_cross and not short_cross:
                continue
            # 🧠 ML Signal: Differentiating logic based on trade direction

            # Push order status update for filled orders
            order.traded = order.volume
            order.status = Status.ALLTRADED
            self.strategy.update_order(order)

            if order.vt_orderid in self.active_limit_orders:
                self.active_limit_orders.pop(order.vt_orderid)

            # 🧠 ML Signal: Calculation of turnover based on trade volume and price
            # Generate trade information
            self.trade_count += 1
            # 🧠 ML Signal: Calculation of trading PnL based on position change and price difference

            if long_cross:
                # 🧠 ML Signal: Accumulating turnover for all trades
                # ✅ Best Practice: Use of type hints for function parameters and return type improves code readability and maintainability.
                trade_price = min(order.price, long_best_price)
            else:
                # 🧠 ML Signal: Calculation of commission based on turnover and rate
                # 🧠 ML Signal: Tracking changes to object attributes can be useful for understanding object state changes over time.
                trade_price = max(order.price, short_best_price)

            # 🧠 ML Signal: Calculation of total PnL as sum of trading and holding PnL
            trade: TradeData = TradeData(
                # ✅ Best Practice: Type annotations for parameters and return value improve code readability and maintainability.
                symbol=order.symbol,
                # 🧠 ML Signal: Calculation of net PnL after deducting commission
                exchange=order.exchange,
                # ✅ Best Practice: Type annotations for attributes improve code readability and maintainability.
                orderid=order.orderid,
                tradeid=str(self.trade_count),
                direction=order.direction,
                offset=order.offset,
                price=trade_price,
                volume=order.volume,
                datetime=self.datetime,
                # 🧠 ML Signal: Iterating over dictionary items is a common pattern that can be used to train ML models.
                gateway_name=self.gateway_name,
            )

            # Update available funds
            size: float = self.sizes[trade.vt_symbol]

            trade_turnover: float = trade.price * trade.volume * size

            # 🧠 ML Signal: Method for adding trade data, useful for learning trade patterns
            if trade.direction == Direction.LONG:
                # ✅ Best Practice: Type hinting for 'trade' and return type improves code readability and maintainability
                trade_commission: float = (
                    trade_turnover * self.long_rates[trade.vt_symbol]
                )
            # 🧠 ML Signal: Accessing contract results by trade symbol, indicating a pattern of data organization
            # ✅ Best Practice: Type hinting for 'contract_result' improves code readability and maintainability
            # 🧠 ML Signal: Method call to add trade to contract result, useful for learning trade processing patterns
            else:
                trade_commission = trade_turnover * self.short_rates[trade.vt_symbol]

            if trade.direction == Direction.LONG:
                self.cash -= trade_turnover
            else:
                self.cash += trade_turnover

            self.cash -= trade_commission
            # ✅ Best Practice: Storing input parameters as instance variables can improve code organization and access.

            # Push trade information
            # ✅ Best Practice: Storing input parameters as instance variables can improve code organization and access.
            self.strategy.update_trade(trade)
            # 🧠 ML Signal: Iterating over items in a dictionary is a common pattern.
            # 🧠 ML Signal: Method calls with multiple parameters can indicate complex operations.
            self.trades[trade.vt_tradeid] = trade

    def get_signal(self) -> pl.DataFrame:
        """Get model prediction signal for current time"""
        if not self.datetime:
            self.write_log("尚未开始数据回放，无法加载模型预测值")
            return pl.DataFrame()
        # 🧠 ML Signal: Using dictionary get method with default values is a common pattern.

        dt: datetime = self.datetime.replace(tzinfo=None)
        # ⚠️ SAST Risk (Low): Potential KeyError if vt_symbol is not in sizes.
        signal: pl.DataFrame = self.signal_df.filter(pl.col("datetime") == dt)

        # ⚠️ SAST Risk (Low): Potential KeyError if vt_symbol is not in long_rates.
        if signal.is_empty():
            self.write_log(f"找不到{dt}对应的信号模型预测值")
        # ⚠️ SAST Risk (Low): Potential KeyError if vt_symbol is not in short_rates.

        return signal

    # 🧠 ML Signal: Accumulating values in a loop is a common pattern.
    def send_order(
        # ✅ Best Practice: Use of dictionary update method for merging dictionaries
        self,
        # 🧠 ML Signal: Accumulating values in a loop is a common pattern.
        strategy: AlphaStrategy,
        # 🧠 ML Signal: Iterating over dictionary items to process key-value pairs
        vt_symbol: str,
        # 🧠 ML Signal: Accumulating values in a loop is a common pattern.
        direction: Direction,
        # 🧠 ML Signal: Accumulating values in a loop is a common pattern.
        # 🧠 ML Signal: Use of type hinting for variable assignment
        offset: Offset,
        price: float,
        # 🧠 ML Signal: Accumulating values in a loop is a common pattern.
        # ✅ Best Practice: Storing results in a dictionary for easy access and modification.
        # 🧠 ML Signal: Conditional logic to handle existing and new entries
        # 🧠 ML Signal: Creating new instances when a condition is not met
        volume: float,
    ) -> list[str]:
        """Send order"""
        price = round_to(price, self.priceticks[vt_symbol])
        symbol, exchange = extract_vt_symbol(vt_symbol)

        self.limit_order_count += 1

        order: OrderData = OrderData(
            symbol=symbol,
            exchange=exchange,
            orderid=str(self.limit_order_count),
            direction=direction,
            offset=offset,
            price=price,
            volume=volume,
            status=Status.SUBMITTING,
            datetime=self.datetime,
            gateway_name=self.gateway_name,
        )

        self.active_limit_orders[order.vt_orderid] = order
        self.limit_orders[order.vt_orderid] = order

        return [order.vt_orderid]

    def cancel_order(self, strategy: AlphaStrategy, vt_orderid: str) -> None:
        """Cancel order"""
        if vt_orderid not in self.active_limit_orders:
            return
        order: OrderData = self.active_limit_orders.pop(vt_orderid)

        order.status = Status.CANCELLED
        self.strategy.update_order(order)

    def write_log(self, msg: str, strategy: AlphaStrategy | None = None) -> None:
        """Output log message"""
        msg = f"{self.datetime}  {msg}"
        self.logs.append(msg)

    def get_all_trades(self) -> list[TradeData]:
        """Get all trade information"""
        return list(self.trades.values())

    def get_all_orders(self) -> list[OrderData]:
        """Get all order information"""
        return list(self.limit_orders.values())

    def get_all_daily_results(self) -> list["PortfolioDailyResult"]:
        """Get all daily profit and loss information"""
        return list(self.daily_results.values())

    def get_cash_available(self) -> float:
        """Get current available cash"""
        return self.cash

    def get_holding_value(self) -> float:
        """Get current holding market value"""
        holding_value: float = 0

        for vt_symbol, pos in self.strategy.pos_data.items():
            bar: BarData = self.bars[vt_symbol]
            size: float = self.sizes[vt_symbol]

            holding_value += bar.close_price * pos * size

        return holding_value


class ContractDailyResult:
    """Contract daily profit and loss result"""

    def __init__(self, result_date: date, close_price: float) -> None:
        """Constructor"""
        self.date: date = result_date
        self.close_price: float = close_price
        self.pre_close: float = 0

        self.trades: list[TradeData] = []
        self.trade_count: int = 0

        self.start_pos: float = 0
        self.end_pos: float = 0

        self.turnover: float = 0
        self.commission: float = 0

        self.trading_pnl: float = 0
        self.holding_pnl: float = 0
        self.total_pnl: float = 0
        self.net_pnl: float = 0

    def add_trade(self, trade: TradeData) -> None:
        """Add trade information"""
        self.trades.append(trade)

    def calculate_pnl(
        self,
        pre_close: float,
        start_pos: float,
        size: float,
        long_rate: float,
        short_rate: float,
    ) -> None:
        """Calculate profit and loss"""
        # If there is no previous close price, use 1 instead to avoid division error
        if pre_close:
            self.pre_close = pre_close
        # else:
        #     self.pre_close = 1

        # Calculate holding profit and loss
        self.start_pos = start_pos
        self.end_pos = start_pos

        self.holding_pnl = self.start_pos * (self.close_price - self.pre_close) * size

        # Calculate trading profit and loss
        self.trade_count = len(self.trades)

        for trade in self.trades:
            if trade.direction == Direction.LONG:
                pos_change: float = trade.volume
                rate: float = long_rate
            else:
                pos_change = -trade.volume
                rate = short_rate

            self.end_pos += pos_change

            turnover: float = trade.volume * size * trade.price

            self.trading_pnl += pos_change * (self.close_price - trade.price) * size
            self.turnover += turnover
            self.commission += turnover * rate

        # Calculate daily profit and loss
        self.total_pnl = self.trading_pnl + self.holding_pnl
        self.net_pnl = self.total_pnl - self.commission

    def update_close_price(self, close_price: float) -> None:
        """Update daily close price"""
        self.close_price = close_price


class PortfolioDailyResult:
    """Portfolio daily profit and loss result"""

    def __init__(self, result_date: date, close_prices: dict[str, float]) -> None:
        """Constructor"""
        self.date: date = result_date
        self.close_prices: dict[str, float] = close_prices
        self.pre_closes: dict[str, float] = {}
        self.start_poses: dict[str, float] = {}
        self.end_poses: dict[str, float] = {}

        self.contract_results: dict[str, ContractDailyResult] = {}

        for vt_symbol, close_price in close_prices.items():
            self.contract_results[vt_symbol] = ContractDailyResult(
                result_date, close_price
            )

        self.trade_count: int = 0
        self.turnover: float = 0
        self.commission: float = 0
        self.trading_pnl: float = 0
        self.holding_pnl: float = 0
        self.total_pnl: float = 0
        self.net_pnl: float = 0

    def add_trade(self, trade: TradeData) -> None:
        """Add trade information"""
        contract_result: ContractDailyResult = self.contract_results[trade.vt_symbol]
        contract_result.add_trade(trade)

    def calculate_pnl(
        self,
        pre_closes: dict[str, float],
        start_poses: dict[str, float],
        sizes: dict[str, float],
        long_rates: dict[str, float],
        short_rates: dict[str, float],
    ) -> None:
        """Calculate profit and loss"""
        self.pre_closes = pre_closes
        self.start_poses = start_poses

        for vt_symbol, contract_result in self.contract_results.items():
            contract_result.calculate_pnl(
                pre_closes.get(vt_symbol, 0),
                start_poses.get(vt_symbol, 0),
                sizes[vt_symbol],
                long_rates[vt_symbol],
                short_rates[vt_symbol],
            )

            self.trade_count += contract_result.trade_count
            self.turnover += contract_result.turnover
            self.commission += contract_result.commission
            self.trading_pnl += contract_result.trading_pnl
            self.holding_pnl += contract_result.holding_pnl
            self.total_pnl += contract_result.total_pnl
            self.net_pnl += contract_result.net_pnl

            self.end_poses[vt_symbol] = contract_result.end_pos

    def update_close_prices(self, close_prices: dict[str, float]) -> None:
        """Update daily close prices"""
        self.close_prices.update(close_prices)

        for vt_symbol, close_price in close_prices.items():
            contract_result: ContractDailyResult | None = self.contract_results.get(
                vt_symbol, None
            )
            if contract_result:
                contract_result.update_close_price(close_price)
            else:
                self.contract_results[vt_symbol] = ContractDailyResult(
                    self.date, close_price
                )
