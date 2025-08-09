from collections import defaultdict

# ✅ Best Practice: Grouping imports by standard, third-party, and local can improve readability.
import polars as pl

from vnpy.trader.object import BarData, TradeData
from vnpy.trader.constant import Direction
from vnpy.trader.utility import round_to

# 🧠 ML Signal: Use of class-level attributes for configuration
from vnpy.alpha import AlphaStrategy

# 🧠 ML Signal: Use of class-level attributes for configuration


class EquityDemoStrategy(AlphaStrategy):
    # 🧠 ML Signal: Use of class-level attributes for configuration
    """Equity Long-Only Demo Strategy"""

    # 🧠 ML Signal: Use of class-level attributes for configuration
    top_k: int = 50  # Maximum number of stocks to hold
    n_drop: int = 5  # Number of stocks to sell each time
    # 🧠 ML Signal: Use of class-level attributes for configuration
    min_days: int = 3  # Minimum holding period in days
    cash_ratio: float = 0.95  # Cash utilization ratio
    # 🧠 ML Signal: Use of class-level attributes for configuration
    min_volume: int = 100  # Minimum trading unit
    # ✅ Best Practice: Type hinting for better code readability and maintainability
    open_rate: float = 0.0005  # Opening commission rate
    # 🧠 ML Signal: Use of class-level attributes for configuration
    close_rate: float = 0.0015  # Closing commission rate
    # 🧠 ML Signal: Logging initialization events can be useful for tracking system behavior
    min_commission: int = 5  # Minimum commission value
    # 🧠 ML Signal: Use of class-level attributes for configuration
    price_add: float = 0.05  # Order price adjustment ratio
    # 🧠 ML Signal: Use of callback function pattern

    # 🧠 ML Signal: Use of class-level attributes for configuration
    def on_init(self) -> None:
        # ⚠️ SAST Risk (Low): Potential KeyError if vt_symbol is not in holding_days
        """Strategy initialization callback"""
        # ✅ Best Practice: Use of pop with default value to avoid KeyError
        # Dictionary to track stock holding days
        # 🧠 ML Signal: Usage of a method to get the last signal, indicating a pattern of retrieving and processing signals
        self.holding_days: defaultdict = defaultdict(int)

        # ✅ Best Practice: Sorting data for prioritized processing
        self.write_log("Strategy initialized")

    # 🧠 ML Signal: Extracting symbols with positions, indicating a pattern of managing active trades
    def on_trade(self, trade: TradeData) -> None:
        """Trade execution callback"""
        # Remove holding days record when selling
        # 🧠 ML Signal: Incrementing holding days, indicating a pattern of tracking trade duration
        if trade.direction == Direction.SHORT:
            self.holding_days.pop(trade.vt_symbol, None)

    # 🧠 ML Signal: Determining active symbols, indicating a pattern of selecting top signals

    def on_bars(self, bars: dict[str, BarData]) -> None:
        """K-line slice callback"""
        # ✅ Best Practice: Filtering data for relevant symbols
        # Get the latest signals and sort them
        last_signal: pl.DataFrame = self.get_signal()
        # 🧠 ML Signal: Identifying component symbols, indicating a pattern of managing portfolio components
        last_signal = last_signal.sort("signal", descending=True)

        # 🧠 ML Signal: Determining symbols to sell, indicating a pattern of managing exits
        # Get position symbols and update holding days
        pos_symbols: list[str] = [
            vt_symbol for vt_symbol, pos in self.pos_data.items() if pos
        ]

        for vt_symbol in pos_symbols:
            self.holding_days[vt_symbol] += 1
        # ✅ Best Practice: Filtering data for buyable symbols

        # Generate sell list
        # 🧠 ML Signal: Calculating buy quantity, indicating a pattern of managing entry sizes
        active_symbols: set[str] = set(
            last_signal["vt_symbol"][: self.top_k]
        )  # Extract symbols with highest signals
        active_symbols.update(pos_symbols)  # Merge with currently held symbols
        # 🧠 ML Signal: Selecting symbols to buy, indicating a pattern of managing entries
        active_df: pl.DataFrame = last_signal.filter(
            pl.col("vt_symbol").is_in(active_symbols)
        )  # Filter signals for these symbols

        # 🧠 ML Signal: Retrieving available cash, indicating a pattern of managing capital
        component_symbols: set[str] = set(
            last_signal["vt_symbol"]
        )  # Extract current index component symbols
        sell_symbols: set[str] = set(pos_symbols).difference(
            component_symbols
        )  # Sell positions not in components

        for vt_symbol in active_df["vt_symbol"][
            -self.n_drop :
        ]:  # Iterate through lowest signal portion
            if vt_symbol in pos_symbols:  # If the contract is in current positions
                # ⚠️ SAST Risk (Low): Potential NoneType access if bar is None
                sell_symbols.add(vt_symbol)  # Add it to sell list

        # Generate buy list
        buyable_df: pl.DataFrame = last_signal.filter(
            ~pl.col("vt_symbol").is_in(pos_symbols)
        )  # Filter contracts available for purchase
        # 🧠 ML Signal: Calculating sell price and volume, indicating a pattern of executing trades
        buy_quantity: int = (
            len(sell_symbols) + self.top_k - len(pos_symbols)
        )  # Calculate number of contracts to buy
        # 🧠 ML Signal: Setting target to zero, indicating a pattern of closing positions
        # 🧠 ML Signal: Calculating turnover and cost, indicating a pattern of managing trade costs
        # 🧠 ML Signal: Calculating buy value per symbol, indicating a pattern of allocating capital
        # ⚠️ SAST Risk (Low): Potential KeyError if vt_symbol not in bars
        # 🧠 ML Signal: Calculating buy volume, indicating a pattern of determining trade size
        # 🧠 ML Signal: Executing trades, indicating a pattern of finalizing trade actions
        buy_symbols: list = list(
            buyable_df[:buy_quantity]["vt_symbol"]
        )  # Select buy contract code list

        # Sell rebalancing
        cash: float = (
            self.get_cash_available()
        )  # Get available cash after yesterday's settlement

        for vt_symbol in sell_symbols:
            if (
                self.holding_days[vt_symbol] < self.min_days
            ):  # Check if holding period exceeds threshold
                continue

            bar: BarData | None = bars.get(
                vt_symbol
            )  # Get current price of the contract
            if not bar:
                continue
            sell_price: float = bar.close_price

            sell_volume: float = self.get_pos(vt_symbol)  # Get current holding volume

            self.set_target(vt_symbol, target=0)  # Set target volume to 0

            turnover: float = sell_price * sell_volume  # Calculate selling turnover
            cost: float = max(
                turnover * self.close_rate, self.min_commission
            )  # Calculate selling cost
            cash += turnover - cost  # Update available cash

        # Buy rebalancing
        if buy_symbols:
            buy_value: float = (
                cash * self.cash_ratio / len(buy_symbols)
            )  # Calculate investment amount per contract

            for vt_symbol in buy_symbols:
                buy_price: float = bars[
                    vt_symbol
                ].close_price  # Get current price of the contract
                if not buy_price:
                    continue

                buy_volume: float = round_to(
                    buy_value / buy_price, self.min_volume
                )  # Calculate volume to buy

                self.set_target(vt_symbol, buy_volume)  # Set target holding volume

        # Execute trading
        self.execute_trading(bars, price_add=self.price_add)
