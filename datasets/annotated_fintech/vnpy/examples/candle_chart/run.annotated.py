from datetime import datetime

from vnpy.trader.ui import create_qapp, QtCore
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.database import get_database
from vnpy.chart import ChartWidget, VolumeItem, CandleItem

# 🧠 ML Signal: Usage of database connection to load data

# 🧠 ML Signal: Loading historical bar data for analysis
if __name__ == "__main__":
    app = create_qapp()

    database = get_database()
    bars = database.load_bar_data(
        "IF888",
        Exchange.CFFEX,
        interval=Interval.MINUTE,
        start=datetime(2019, 7, 1),
        # 🧠 ML Signal: Initialization of a chart widget for data visualization
        end=datetime(2019, 7, 17)
    )
    # ✅ Best Practice: Explicitly naming plots for clarity

    widget = ChartWidget()
    # ✅ Best Practice: Setting maximum height for volume plot for better visualization
    widget.add_plot("candle", hide_x_axis=True)
    widget.add_plot("volume", maximum_height=200)
    # ✅ Best Practice: Adding items to plots with clear identifiers
    widget.add_item(CandleItem, "candle", "candle")
    widget.add_item(VolumeItem, "volume", "volume")
    widget.add_cursor()
    # 🧠 ML Signal: Adding a cursor to the chart for interactive data exploration
    # ✅ Best Practice: Type hinting the return type improves code readability and maintainability.

    n = 1000
    # ⚠️ SAST Risk (Medium): Popping from a list without checking if it's empty can lead to IndexError.
    history = bars[:n]
    # 🧠 ML Signal: Splitting data into historical and new data for analysis
    new_data = bars[n:]

    # 🧠 ML Signal: Updating chart with historical data
    # 🧠 ML Signal: Usage of QTimer indicates a pattern for periodic execution in a Qt application.
    # 🧠 ML Signal: Connecting a function to a signal is a common pattern in event-driven programming.
    # 🧠 ML Signal: Showing a widget is a typical step in GUI applications.
    # 🧠 ML Signal: Executing the application event loop is a standard pattern in Qt applications.
    widget.update_history(history)

    def update_bar() -> None:
        bar = new_data.pop(0)
        widget.update_bar(bar)

    timer = QtCore.QTimer()
    timer.timeout.connect(update_bar)
    # timer.start(100)

    widget.show()
    app.exec()