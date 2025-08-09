from abc import abstractmethod

# ✅ Best Practice: Grouping imports from the same module together improves readability.
import pyqtgraph as pg      # type: ignore

from vnpy.trader.ui import QtCore, QtGui, QtWidgets
# ✅ Best Practice: Grouping imports from the same module together improves readability.
from vnpy.trader.object import BarData
# ✅ Best Practice: Add a class docstring to describe the purpose and usage of the class

from .base import BLACK_COLOR, UP_COLOR, DOWN_COLOR, PEN_WIDTH, BAR_WIDTH
from .manager import BarManager


# 🧠 ML Signal: Usage of type annotations for class attributes
class ChartItem(pg.GraphicsObject):
    """"""
    # 🧠 ML Signal: Usage of type annotations for class attributes

    def __init__(self, manager: BarManager) -> None:
        # 🧠 ML Signal: Usage of type annotations for class attributes
        """"""
        super().__init__()

        # 🧠 ML Signal: Usage of type annotations for class attributes
        self._manager: BarManager = manager
        # 🧠 ML Signal: Usage of type annotations for class attributes

        self._bar_picutures: dict[int, QtGui.QPicture | None] = {}
        self._item_picuture: QtGui.QPicture | None = None

        # 🧠 ML Signal: Usage of type annotations for class attributes
        self._black_brush: QtGui.QBrush = pg.mkBrush(color=BLACK_COLOR)

        # 🧠 ML Signal: Usage of type annotations for class attributes
        self._up_pen: QtGui.QPen = pg.mkPen(
            color=UP_COLOR, width=PEN_WIDTH
        )
        # 🧠 ML Signal: Usage of type annotations for class attributes
        # ✅ Best Practice: Include a docstring to describe the purpose and usage of the method
        self._up_brush: QtGui.QBrush = pg.mkBrush(color=UP_COLOR)

        self._down_pen: QtGui.QPen = pg.mkPen(
            # 🧠 ML Signal: Usage of type annotations for class attributes
            color=DOWN_COLOR, width=PEN_WIDTH
        )
        # ✅ Best Practice: Use of setFlag method to configure item behavior
        self._down_brush: QtGui.QBrush = pg.mkBrush(color=DOWN_COLOR)
        # 🧠 ML Signal: Usage of type annotations for class attributes
        # ✅ Best Practice: Include a docstring to describe the method's purpose
        # ✅ Best Practice: Use of @abstractmethod indicates this method should be implemented by subclasses

        self._rect_area: tuple[float, float] | None = None

        # Very important! Only redraw the visible part and improve speed a lot.
        self.setFlag(self.GraphicsItemFlag.ItemUsesExtendedStyleOption)

        # ✅ Best Practice: Type hints for parameters and return values improve code readability and maintainability.
        # Force update during the next paint
        self._to_update: bool = False

    @abstractmethod
    def _draw_bar_picture(self, ix: int, bar: BarData) -> QtGui.QPicture:
        """
        Draw picture for specific bar.
        # ✅ Best Practice: Include a docstring to describe the method's functionality
        # ✅ Best Practice: Using @abstractmethod indicates that this method should be implemented by subclasses, enforcing a contract for class design.
        """
        pass

    @abstractmethod
    def boundingRect(self) -> QtCore.QRectF:
        """
        Get bounding rectangles for item.
        """
        # ✅ Best Practice: Clear the list before updating to avoid stale data.
        pass

    # 🧠 ML Signal: Usage of a manager pattern to retrieve data.
    @abstractmethod
    def get_y_range(self, min_ix: int | None = None, max_ix: int | None = None) -> tuple[float, float]:
        """
        Get range of y-axis with given x-axis range.

        If min_ix and max_ix not specified, then return range with whole data set.
        """
        # ✅ Best Practice: Type hinting for 'ix' improves code readability and maintainability.
        pass

    # ⚠️ SAST Risk (Low): Potential issue if 'get_index' method does not handle invalid datetime inputs properly.
    @abstractmethod
    def get_info_text(self, ix: int) -> str:
        """
        Get information text to show by cursor.
        # ✅ Best Practice: Calling 'update' method suggests a clear separation of concerns.
        """
        pass

    # ✅ Best Practice: Check if the scene exists before updating to avoid potential errors.
    def update_history(self, history: list[BarData]) -> None:
        """
        Update a list of bar data.
        # 🧠 ML Signal: Calling an update method on a scene object.
        """
        self._bar_picutures.clear()

        bars: list[BarData] = self._manager.get_all_bars()

        for ix, _ in enumerate(bars):
            self._bar_picutures[ix] = None

        self.update()

    def update_bar(self, bar: BarData) -> None:
        """
        Update single bar data.
        """
        # ✅ Best Practice: Use a consistent naming convention for variables (e.g., rect_area instead of rect_area: tuple).
        ix: int | None = self._manager.get_index(bar.datetime)
        if ix is None:
            return

        self._bar_picutures[ix] = None

        self.update()

    def update(self) -> None:
        """
        Refresh the item.
        """
        # ⚠️ SAST Risk (Low): Potential typo in attribute name _item_picuture, which may lead to runtime errors.
        if self.scene():
            self._to_update = True
            self.scene().update()
    # ✅ Best Practice: Initialize _item_picuture to ensure it's always set before use

    def paint(
        # ✅ Best Practice: Type hinting for painter improves code readability and maintainability
        self,
        painter: QtGui.QPainter,
        opt: QtWidgets.QStyleOptionGraphicsItem,
        # ✅ Best Practice: Type hinting for bar_picture improves code readability and maintainability
        w: QtWidgets.QWidget
    ) -> None:
        """
        Reimplement the paint method of parent class.

        This function is called by external QGraphicsView.
        """
        # 🧠 ML Signal: Caching pattern with self._bar_picutures can be used to train models on optimization techniques
        rect: QtCore.QRectF = opt.exposedRect       # type: ignore

        min_ix: int = int(rect.left())
        max_ix: int = int(rect.right())
        # ⚠️ SAST Risk (Low): Potential for NoneType error if bar_picture is None, though unlikely due to prior checks
        # ✅ Best Practice: Consider adding type hints for attributes like _item_picuture and _bar_picutures for better readability and maintainability.
        max_ix = min(max_ix, len(self._bar_picutures))
        # ✅ Best Practice: Ensure painter is properly ended to release resources

        # ✅ Best Practice: Ensure that _bar_picutures is initialized as a list or similar collection to avoid AttributeError.
        rect_area: tuple = (min_ix, max_ix)
        # ✅ Best Practice: Class docstring is empty; consider providing a description of the class.
        if (
            # 🧠 ML Signal: Method call pattern on self object, useful for understanding object behavior.
            self._to_update
            or rect_area != self._rect_area
            or not self._item_picuture
        # ✅ Best Practice: Use of type hinting for the 'manager' parameter improves code readability and maintainability.
        ):
            self._to_update = False
            # ✅ Best Practice: Calling the superclass's __init__ method ensures proper initialization of the base class.
            self._rect_area = rect_area
            # ✅ Best Practice: Type hinting improves code readability and maintainability
            self._draw_item_picture(min_ix, max_ix)

        # ✅ Best Practice: Type hinting improves code readability and maintainability
        if self._item_picuture:
            self._item_picuture.play(painter)
    # 🧠 ML Signal: Conditional logic based on price comparison

    def _draw_item_picture(self, min_ix: int, max_ix: int) -> None:
        """
        Draw the picture of item in specific range.
        """
        # 🧠 ML Signal: Conditional logic based on price comparison
        self._item_picuture = QtGui.QPicture()
        painter: QtGui.QPainter = QtGui.QPainter(self._item_picuture)

        for ix in range(min_ix, max_ix):
            # 🧠 ML Signal: Drawing logic based on price range
            bar_picture: QtGui.QPicture | None = self._bar_picutures[ix]

            if bar_picture is None:
                bar: BarData | None = self._manager.get_bar(ix)
                if bar is None:
                    continue
                # 🧠 ML Signal: Handling special case where open and close prices are equal

                bar_picture = self._draw_bar_picture(ix, bar)
                self._bar_picutures[ix] = bar_picture

            bar_picture.play(painter)

        # 🧠 ML Signal: Drawing rectangle for price difference
        painter.end()
    # ✅ Best Practice: Type hinting improves code readability and maintainability

    def clear_all(self) -> None:
        """
        Clear all data in the item.
        """
        # 🧠 ML Signal: Usage of a method to get a price range, indicating a pattern of data retrieval
        # ✅ Best Practice: Type hinting for variable 'rect' improves code readability and maintainability
        self._item_picuture = None
        self._bar_picutures.clear()
        self.update()


class CandleItem(ChartItem):
    # 🧠 ML Signal: Accessing a property '_bar_picutures' suggests a pattern of using class attributes
    """"""

    # ✅ Best Practice: Docstring provides a clear explanation of the method's purpose and behavior.
    # 🧠 ML Signal: Returning a constructed object, indicating a pattern of object creation and return
    def __init__(self, manager: BarManager) -> None:
        """"""
        super().__init__(manager)

    def _draw_bar_picture(self, ix: int, bar: BarData) -> QtGui.QPicture:
        # 🧠 ML Signal: Usage of a method to get a range of values, which could be a common pattern in data processing.
        """"""
        # Create objects
        # ✅ Best Practice: Returning a tuple directly is a clear and concise way to return multiple values.
        candle_picture: QtGui.QPicture = QtGui.QPicture()
        painter: QtGui.QPainter = QtGui.QPainter(candle_picture)

        # 🧠 ML Signal: Usage of type hinting for variables and return types
        # Set painter color
        # ✅ Best Practice: Use of type hinting for better code readability and maintainability
        if bar.close_price >= bar.open_price:
            # ✅ Best Practice: Use of list to accumulate strings for better performance and readability
            # ✅ Best Practice: Use of strftime for date formatting
            painter.setPen(self._up_pen)
            painter.setBrush(self._black_brush)
        else:
            painter.setPen(self._down_pen)
            painter.setBrush(self._down_brush)

        # Draw candle shadow
        if bar.high_price > bar.low_price:
            painter.drawLine(
                QtCore.QPointF(ix, bar.high_price),
                QtCore.QPointF(ix, bar.low_price)
            )

        # Draw candle body
        if bar.open_price == bar.close_price:
            painter.drawLine(
                QtCore.QPointF(ix - BAR_WIDTH, bar.open_price),
                QtCore.QPointF(ix + BAR_WIDTH, bar.open_price),
            )
        # ✅ Best Practice: Explicit conversion of numbers to strings
        else:
            rect: QtCore.QRectF = QtCore.QRectF(
                # ✅ Best Practice: Explicit conversion of numbers to strings
                ix - BAR_WIDTH,
                bar.open_price,
                BAR_WIDTH * 2,
                bar.close_price - bar.open_price
            # ✅ Best Practice: Add a class docstring to describe the purpose and usage of the class
            # ✅ Best Practice: Explicit conversion of numbers to strings
            # 🧠 ML Signal: Constructor method indicating object initialization pattern
            )
            painter.drawRect(rect)

        # ✅ Best Practice: Use of join for efficient string concatenation
        # ✅ Best Practice: Calling the superclass's constructor to ensure proper initialization
        # Finish
        painter.end()
        # ✅ Best Practice: Type annotations for variables improve code readability and maintainability.
        return candle_picture

    # ✅ Best Practice: Type annotations for variables improve code readability and maintainability.
    def boundingRect(self) -> QtCore.QRectF:
        """"""
        # 🧠 ML Signal: Conditional logic based on object attributes can indicate decision-making patterns.
        min_price, max_price = self._manager.get_price_range()
        rect: QtCore.QRectF = QtCore.QRectF(
            0,
            min_price,
            # 🧠 ML Signal: Conditional logic based on object attributes can indicate decision-making patterns.
            len(self._bar_picutures),
            max_price - min_price
        )
        return rect

    def get_y_range(self, min_ix: int | None = None, max_ix: int | None = None) -> tuple[float, float]:
        """
        Get range of y-axis with given x-axis range.

        If min_ix and max_ix not specified, then return range with whole data set.
        """
        # 🧠 ML Signal: Drawing operations can be indicative of graphical rendering patterns.
        min_price, max_price = self._manager.get_price_range(min_ix, max_ix)
        # 🧠 ML Signal: Usage of a method to get a range, indicating a pattern of data retrieval
        # ✅ Best Practice: Type hinting for variable 'rect' improves code readability and maintainability
        return min_price, max_price

    def get_info_text(self, ix: int) -> str:
        """
        Get information text to show by cursor.
        """
        # 🧠 ML Signal: Accessing a length property, indicating a pattern of collection size usage
        bar: BarData | None = self._manager.get_bar(ix)

        # 🧠 ML Signal: Returning a constructed object, indicating a pattern of object creation
        # ✅ Best Practice: Type hinting improves code readability and maintainability.
        if bar:
            words: list = [
                "Date",
                bar.datetime.strftime("%Y-%m-%d"),
                "",
                # 🧠 ML Signal: Method usage patterns can be used to understand how range queries are performed.
                "Time",
                bar.datetime.strftime("%H:%M"),
                # 🧠 ML Signal: Return value patterns can be used to understand typical output ranges.
                "",
                "Open",
                str(bar.open_price),
                # ✅ Best Practice: Type hinting improves code readability and maintainability.
                "",
                "High",
                # 🧠 ML Signal: Checking if a variable is None before using it is a common pattern.
                str(bar.high_price),
                "",
                # 🧠 ML Signal: String formatting with f-strings is a common pattern.
                "Low",
                # 🧠 ML Signal: Returning a string based on a condition is a common pattern.
                str(bar.low_price),
                "",
                "Close",
                str(bar.close_price)
            ]
            text: str = "\n".join(words)
        else:
            text = ""

        return text


class VolumeItem(ChartItem):
    """"""

    def __init__(self, manager: BarManager) -> None:
        """"""
        super().__init__(manager)

    def _draw_bar_picture(self, ix: int, bar: BarData) -> QtGui.QPicture:
        """"""
        # Create objects
        volume_picture: QtGui.QPicture = QtGui.QPicture()
        painter: QtGui.QPainter = QtGui.QPainter(volume_picture)

        # Set painter color
        if bar.close_price >= bar.open_price:
            painter.setPen(self._up_pen)
            painter.setBrush(self._up_brush)
        else:
            painter.setPen(self._down_pen)
            painter.setBrush(self._down_brush)

        # Draw volume body
        rect: QtCore.QRectF = QtCore.QRectF(
            ix - BAR_WIDTH,
            0,
            BAR_WIDTH * 2,
            bar.volume
        )
        painter.drawRect(rect)

        # Finish
        painter.end()
        return volume_picture

    def boundingRect(self) -> QtCore.QRectF:
        """"""
        min_volume, max_volume = self._manager.get_volume_range()
        rect: QtCore.QRectF = QtCore.QRectF(
            0,
            min_volume,
            len(self._bar_picutures),
            max_volume - min_volume
        )
        return rect

    def get_y_range(self, min_ix: int | None = None, max_ix: int | None = None) -> tuple[float, float]:
        """
        Get range of y-axis with given x-axis range.

        If min_ix and max_ix not specified, then return range with whole data set.
        """
        min_volume, max_volume = self._manager.get_volume_range(min_ix, max_ix)
        return min_volume, max_volume

    def get_info_text(self, ix: int) -> str:
        """
        Get information text to show by cursor.
        """
        bar: BarData | None = self._manager.get_bar(ix)

        if bar:
            text: str = f"Volume {bar.volume}"
        else:
            text = ""

        return text