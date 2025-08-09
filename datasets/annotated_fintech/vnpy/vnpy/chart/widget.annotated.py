from datetime import datetime

# ✅ Best Practice: Grouping related imports together improves readability.
import pyqtgraph as pg  # type: ignore

from vnpy.trader.ui import QtGui, QtWidgets, QtCore
from vnpy.trader.object import BarData

from .manager import BarManager
from .base import (
    GREY_COLOR,
    WHITE_COLOR,
    CURSOR_COLOR,
    BLACK_COLOR,
    to_int,
    NORMAL_FONT,
)

# ✅ Best Practice: Setting configuration options at the start of the script is a good practice for clarity.
# ✅ Best Practice: Class docstring is empty; consider providing a description of the class.
from .axis import DatetimeAxis
from .item import ChartItem

# ✅ Best Practice: Constants should be documented or named descriptively to convey their purpose.

pg.setConfigOptions(antialias=True)

# ✅ Best Practice: Initialize all instance variables in the constructor for clarity and maintainability


class ChartWidget(pg.PlotWidget):
    # ✅ Best Practice: Use type annotations for instance variables for better readability and type checking
    """"""
    MIN_BAR_COUNT = 100

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """"""
        super().__init__(parent)

        self._manager: BarManager = BarManager()
        # ✅ Best Practice: Encapsulate UI initialization in a separate method for better organization

        # ✅ Best Practice: Set a descriptive window title for better user experience
        self._plots: dict[str, pg.PlotItem] = {}
        self._items: dict[str, ChartItem] = {}
        # 🧠 ML Signal: Usage of a specific graphics layout library (pg.GraphicsLayout)
        self._item_plot_map: dict[ChartItem, pg.PlotItem] = {}

        # ✅ Best Practice: Set consistent margins for layout aesthetics
        self._first_plot: pg.PlotItem | None = None
        self._cursor: ChartCursor | None = None
        # ✅ Best Practice: Set spacing to 0 for a compact layout

        # ✅ Best Practice: Use of type hint for return value improves code readability and maintainability
        self._right_ix: int = 0  # Index of most right data
        # ✅ Best Practice: Set border color and width for visual clarity
        self._bar_count: int = self.MIN_BAR_COUNT  # Total bar visible in chart
        # 🧠 ML Signal: Method returns a new instance of a class, indicating a factory or builder pattern

        # ✅ Best Practice: Set Z value to control stacking order of items
        self._init_ui()

    # 🧠 ML Signal: Checks for the existence of an attribute before assignment

    # 🧠 ML Signal: Instantiation of an object with multiple dependencies
    # ✅ Best Practice: Set central item to ensure layout is displayed
    def _init_ui(self) -> None:
        """"""
        self.setWindowTitle("ChartWidget of VeighNa")

        self._layout: pg.GraphicsLayout = pg.GraphicsLayout()
        self._layout.setContentsMargins(10, 10, 10, 10)
        self._layout.setSpacing(0)
        self._layout.setBorder(color=GREY_COLOR, width=0.8)
        self._layout.setZValue(0)
        self.setCentralItem(self._layout)

    def _get_new_x_axis(self) -> DatetimeAxis:
        # ✅ Best Practice: Type hinting for plot variable improves code readability and maintainability
        return DatetimeAxis(self._manager, orientation="bottom")

    def add_cursor(self) -> None:
        """"""
        if not self._cursor:
            self._cursor = ChartCursor(
                self, self._manager, self._plots, self._item_plot_map
            )

    def add_plot(
        self,
        plot_name: str,
        minimum_height: int = 80,
        maximum_height: int | None = None,
        hide_x_axis: bool = False,
    ) -> None:
        """
        Add plot area.
        """
        # 🧠 ML Signal: Usage of signal-slot connection pattern
        # Create plot object
        plot: pg.PlotItem = pg.PlotItem(axisItems={"bottom": self._get_new_x_axis()})
        plot.setMenuEnabled(False)
        # ✅ Best Practice: Type hinting for right_axis variable improves code readability and maintainability
        plot.setClipToView(True)
        plot.hideAxis("left")
        plot.showAxis("right")
        plot.setDownsampling(mode="peak")
        plot.setRange(xRange=(0, 1), yRange=(0, 1))
        # ✅ Best Practice: Type hinting for first_plot variable improves code readability and maintainability
        plot.hideButtons()
        # 🧠 ML Signal: Usage of dictionary to store plot objects
        plot.setMinimumHeight(minimum_height)

        if maximum_height:
            plot.setMaximumHeight(maximum_height)

        if hide_x_axis:
            plot.hideAxis("bottom")

        if not self._first_plot:
            # 🧠 ML Signal: Instantiating objects using a class passed as a parameter
            self._first_plot = plot

        # ✅ Best Practice: Using type annotations for dictionary keys and values
        # Connect view change signal to update y range function
        view: pg.ViewBox = plot.getViewBox()
        # ⚠️ SAST Risk (Low): Potential NoneType error if plot_name is not found in self._plots
        view.sigXRangeChanged.connect(self._update_y_range)
        # ✅ Best Practice: Include type hints for method parameters and return type for better readability and maintainability
        view.setMouseEnabled(x=True, y=False)
        # ⚠️ SAST Risk (Low): Possible AttributeError if plot is None

        # Set right axis
        right_axis: pg.AxisItem = plot.getAxis("right")
        # ✅ Best Practice: Maintaining a mapping between items and plots for easy reference
        right_axis.setWidth(60)
        # 🧠 ML Signal: Usage of dictionary get method with default value
        # ✅ Best Practice: Type hinting for return value improves code readability and maintainability
        right_axis.tickFont = NORMAL_FONT

        # Connect x-axis link
        if self._plots:
            first_plot: pg.PlotItem = list(self._plots.values())[0]
            # 🧠 ML Signal: Accessing instance variables like self._plots can indicate object-oriented design patterns
            plot.setXLink(first_plot)

        # Store plot object in dict
        self._plots[plot_name] = plot
        # ✅ Best Practice: Clear and concise docstring explaining the method's purpose.

        # Add plot onto the layout
        # 🧠 ML Signal: Method chaining pattern, indicating a sequence of operations.
        self._layout.nextRow()
        self._layout.addItem(plot)

    # 🧠 ML Signal: Iterating over a collection to perform an operation on each item.

    def add_item(
        # 🧠 ML Signal: Recursively calling a method on items in a collection.
        self,
        item_class: type[ChartItem],
        item_name: str,
        # ⚠️ SAST Risk (Low): Potential for NoneType error if _cursor is not properly checked.
        # ✅ Best Practice: Use of type hints for function parameters and return type improves code readability and maintainability.
        plot_name: str,
        # 🧠 ML Signal: Conditional operation based on the presence of an attribute.
    ) -> None:
        """
        Add chart item.
        # 🧠 ML Signal: Calling a method on each item in a collection, indicating a pattern of updating or processing multiple objects.
        """
        item: ChartItem = item_class(self._manager)
        # 🧠 ML Signal: Method call to update plot limits, indicating a pattern of visual data representation.
        self._items[item_name] = item

        plot: pg.PlotItem = self._plots.get(plot_name)
        # 🧠 ML Signal: Method updates internal state based on input data
        # 🧠 ML Signal: Method call to adjust view or position, indicating a pattern of user interface or visualization adjustment.
        plot.addItem(item)

        # 🧠 ML Signal: Iterating over a collection to update each item
        self._item_plot_map[item] = plot

    # 🧠 ML Signal: Method updates internal state based on input data
    def get_plot(self, plot_name: str) -> pg.PlotItem:
        """
        Get specific plot with its name.
        # ✅ Best Practice: Use parentheses for clarity in complex conditions
        """
        return self._plots.get(plot_name, None)

    # 🧠 ML Signal: Conditional logic triggers a state change
    # 🧠 ML Signal: Iterating over a dictionary to update associated values
    def get_all_plots(self) -> list[pg.PlotItem]:
        """
        Get all plot objects.
        """
        return list(self._plots.values())

    def clear_all(self) -> None:
        """
        Clear all data.
        """
        self._manager.clear_all()

        # ✅ Best Practice: Type hinting for variables improves code readability and maintainability.
        for item in self._items.values():
            item.clear_all()
        # ✅ Best Practice: Type hinting for variables improves code readability and maintainability.

        if self._cursor:
            # 🧠 ML Signal: Iterating over a collection of plots to update their properties.
            self._cursor.clear_all()

    # 🧠 ML Signal: Method call to set a specific range on a plot, indicating usage of a plotting library.

    def update_history(self, history: list[BarData]) -> None:
        """
        Update a list of bar data.
        """
        self._manager.update_history(history)
        # 🧠 ML Signal: Usage of type hinting for variable 'view' indicates a pattern for type-aware programming.

        for item in self._items.values():
            # 🧠 ML Signal: Usage of type hinting for variable 'view_range' indicates a pattern for type-aware programming.
            item.update_history(history)

        # ✅ Best Practice: Use of max function ensures 'min_ix' is non-negative, preventing potential index errors.
        self._update_plot_limits()

        # ⚠️ SAST Risk (Low): Potential risk if 'view_range[0][1]' is not within expected bounds of 'self._manager.get_count()'.
        self.move_to_right()

    # 🧠 ML Signal: Iterating over dictionary items is a common pattern for processing key-value pairs.

    def update_bar(self, bar: BarData) -> None:
        """
        Update single bar data.
        """
        # ✅ Best Practice: Explicitly naming the parameter 'yRange' improves code readability.
        self._manager.update_bar(bar)
        # 🧠 ML Signal: Type hinting for variables can be used to infer expected data types.

        for item in self._items.values():
            # 🧠 ML Signal: Usage of list to store view range values.
            item.update_bar(bar)

        # ⚠️ SAST Risk (Low): Potential risk if view_range[0][1] is not a number or is None.
        self._update_plot_limits()

        if self._right_ix >= (self._manager.get_count() - self._bar_count / 2):
            # ✅ Best Practice: Use of QtCore.Qt.Key for readability and maintainability
            # ✅ Best Practice: Calling the superclass method ensures that the base functionality is preserved.
            self.move_to_right()

    # 🧠 ML Signal: Pattern of handling key press events
    def _update_plot_limits(self) -> None:
        """
        Update the limit of plots.
        """
        # 🧠 ML Signal: Pattern of handling key press events
        for item, plot in self._item_plot_map.items():
            min_value, max_value = item.get_y_range()
            # 🧠 ML Signal: Custom handling for right key press

            plot.setLimits(
                # 🧠 ML Signal: Pattern of handling key press events
                xMin=-1,
                xMax=self._manager.get_count(),
                yMin=min_value,
                # ✅ Best Practice: Use of type annotation for variable 'delta' improves code readability and maintainability.
                # 🧠 ML Signal: Custom handling for up key press
                yMax=max_value,
                # 🧠 ML Signal: Pattern of handling key press events
            )

    # 🧠 ML Signal: Conditional logic based on event data can indicate user interaction patterns.

    # 🧠 ML Signal: Custom handling for down key press
    def _update_x_range(self) -> None:
        """
        Update the x-axis range of plots.
        # 🧠 ML Signal: Conditional logic based on event data can indicate user interaction patterns.
        """
        max_ix: int = self._right_ix
        min_ix: int = self._right_ix - self._bar_count
        # 🧠 ML Signal: Method calls based on user input can be used to understand feature usage.
        # ✅ Best Practice: Use of descriptive method name and docstring for clarity

        for plot in self._plots.values():
            # ✅ Best Practice: Use of max function to ensure _right_ix does not go below _bar_count
            plot.setRange(xRange=(min_ix, max_ix), padding=0)

    # 🧠 ML Signal: Method call pattern for updating UI components
    def _update_y_range(self) -> None:
        """
        Update the y-axis range of plots.
        # 🧠 ML Signal: Conditional logic pattern for object manipulation
        """
        if not self._first_plot:
            return
        # 🧠 ML Signal: Method call pattern for updating UI components
        # 🧠 ML Signal: Method that modifies internal state based on user input

        view: pg.ViewBox = self._first_plot.getViewBox()
        # ✅ Best Practice: Use of min function to ensure _right_ix does not exceed a certain limit
        view_range: list = view.viewRange()

        # 🧠 ML Signal: Method call that updates the display or state
        min_ix: int = max(0, int(view_range[0][0]))
        max_ix: int = min(self._manager.get_count(), int(view_range[0][1]))
        # ✅ Best Practice: Check if _cursor is not None before accessing its methods

        # 🧠 ML Signal: Method that modifies cursor position
        # Update limit for y-axis
        for item, plot in self._item_plot_map.items():
            y_range: tuple = item.get_y_range(min_ix, max_ix)
            # 🧠 ML Signal: Method that updates cursor information
            # 🧠 ML Signal: Method modifies internal state based on user interaction (key down event).
            plot.setRange(yRange=y_range)

    # ✅ Best Practice: Use of min function to ensure _bar_count does not exceed a certain limit.
    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """
        Reimplement this method of parent to update current max_ix value.
        """
        # 🧠 ML Signal: Conditional logic based on the presence of an attribute (self._cursor).
        if not self._first_plot:
            return

        # 🧠 ML Signal: Method updates UI or state based on user interaction.
        # 🧠 ML Signal: Method for handling key up events, indicating user interaction pattern
        view: pg.ViewBox = self._first_plot.getViewBox()
        view_range: list = view.viewRange()
        # ✅ Best Practice: Use of max function to ensure _bar_count does not go below MIN_BAR_COUNT
        self._right_ix = max(0, view_range[0][1])

        # 🧠 ML Signal: Method call to update the x-axis range, indicating a pattern of UI update
        super().paintEvent(event)

    # 🧠 ML Signal: Conditional check for cursor presence, indicating dynamic UI component handling
    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        """
        Reimplement this method of parent to move chart horizontally and zoom in/out.
        # 🧠 ML Signal: Method call to update cursor information, indicating a pattern of UI update
        # ✅ Best Practice: Use of a docstring to describe the method's purpose
        """
        Key = QtCore.Qt.Key
        # 🧠 ML Signal: Method call pattern on an object attribute

        # ⚠️ SAST Risk (Low): Potential risk if get_count() returns unexpected values
        if event.key() == Key.Key_Left:
            self._on_key_left()
        # 🧠 ML Signal: Method call pattern on an object attribute
        elif event.key() == Key.Key_Right:
            # 🧠 ML Signal: Conditional check for attribute existence
            # 🧠 ML Signal: Method call pattern on an object attribute
            # ✅ Best Practice: Consider adding a class docstring to describe the purpose and usage of the class.
            self._on_key_right()
        elif event.key() == Key.Key_Up:
            self._on_key_up()
        elif event.key() == Key.Key_Down:
            self._on_key_down()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        """
        Reimplement this method of parent to zoom in/out.
        """
        # ✅ Best Practice: Type annotations improve code readability and maintainability.
        delta: QtCore.QPoint = event.angleDelta()

        # ✅ Best Practice: Type annotations improve code readability and maintainability.
        if delta.y() > 0:
            self._on_key_up()
        # ✅ Best Practice: Type annotations improve code readability and maintainability.
        elif delta.y() < 0:
            self._on_key_down()

    # ✅ Best Practice: Type annotations improve code readability and maintainability.

    def _on_key_left(self) -> None:
        """
        Move chart to left.
        # ✅ Best Practice: Initialize instance variables with default values.
        # ✅ Best Practice: Consider adding a docstring to describe the purpose of the method
        """
        self._right_ix -= 1
        # ✅ Best Practice: Initialize instance variables with default values.
        # 🧠 ML Signal: Method call pattern indicating UI component initialization
        self._right_ix = max(self._right_ix, self._bar_count)

        # 🧠 ML Signal: Method calls in initialization can indicate component setup patterns.
        # 🧠 ML Signal: Method call pattern indicating UI component initialization
        self._update_x_range()

        if self._cursor:
            # 🧠 ML Signal: Method calls in initialization can indicate component setup patterns.
            # ✅ Best Practice: Use of type annotations for dictionaries improves code readability and maintainability.
            # 🧠 ML Signal: Method call pattern indicating UI component initialization
            self._cursor.move_left()
            self._cursor.update_info()

    def _on_key_right(self) -> None:
        """
        Move chart to right.
        # 🧠 ML Signal: Iterating over items in a dictionary is a common pattern.
        """
        self._right_ix += 1
        # ✅ Best Practice: Use of type annotations for line variables improves code readability and maintainability.
        self._right_ix = min(self._right_ix, self._manager.get_count())

        self._update_x_range()
        # ✅ Best Practice: Use of type annotation for the view variable improves code readability and maintainability.

        if self._cursor:
            # 🧠 ML Signal: Iterating over a list of objects to apply the same operations is a common pattern.
            self._cursor.move_right()
            self._cursor.update_info()

    def _on_key_down(self) -> None:
        """
        Zoom out the chart.
        # 🧠 ML Signal: Storing objects in a dictionary with a key is a common pattern.
        # ✅ Best Practice: Use of type hinting for dictionary keys and values improves code readability and maintainability.
        """
        self._bar_count = int(self._bar_count * 1.2)
        # 🧠 ML Signal: Iterating over dictionary items is a common pattern that can be used to train ML models.
        self._bar_count = min(int(self._bar_count), self._manager.get_count())

        # ✅ Best Practice: Type hinting for variables enhances code readability and maintainability.
        self._update_x_range()

        if self._cursor:
            self._cursor.update_info()

    # ✅ Best Practice: Setting Z-value for graphical items is a good practice for managing rendering order.

    def _on_key_up(self) -> None:
        """
        Zoom in the chart.
        """
        self._bar_count = int(self._bar_count / 1.2)
        # ✅ Best Practice: Type hinting for variables enhances code readability and maintainability.
        self._bar_count = max(int(self._bar_count), self.MIN_BAR_COUNT)

        self._update_x_range()

        # ✅ Best Practice: Setting Z-value for graphical items is a good practice for managing rendering order.
        # ✅ Best Practice: Use of type hinting for dictionary keys and values improves code readability and maintainability.
        if self._cursor:
            self._cursor.update_info()

    # ⚠️ SAST Risk (Low): Ensure that `plot.addItem` does not introduce any side effects or security issues.
    # 🧠 ML Signal: Iterating over dictionary items is a common pattern that can be used to understand data structures.
    # ✅ Best Practice: Type hinting for the variable 'info' improves code readability and maintainability.

    def move_to_right(self) -> None:
        """
        Move chart to the most right.
        """
        self._right_ix = self._manager.get_count()
        # 🧠 ML Signal: Use of constants for colors indicates a pattern for managing UI themes.
        self._update_x_range()

        if self._cursor:
            self._cursor.update_info()


# 🧠 ML Signal: Hiding UI elements initially is a common pattern in UI programming.


# 🧠 ML Signal: Setting Z-value is a pattern for managing UI element layering.
class ChartCursor(QtCore.QObject):
    """"""

    # 🧠 ML Signal: Setting font is a common pattern in UI customization.
    # ✅ Best Practice: Use of docstring to describe the method's purpose
    def __init__(
        # 🧠 ML Signal: Adding items to a plot is a common pattern in data visualization.
        # 🧠 ML Signal: Method name and docstring indicate a pattern of connecting signals to slots
        self,
        # 🧠 ML Signal: Storing UI elements in a dictionary for later access is a common pattern.
        # 🧠 ML Signal: Usage of signal-slot connection pattern in PyQt or PySide
        widget: ChartWidget,
        manager: BarManager,
        plots: dict[str, pg.GraphicsObject],
        # ✅ Best Practice: Check if the manager has any count before proceeding
        item_plot_map: dict[ChartItem, pg.GraphicsObject],
    ) -> None:
        """"""
        # ✅ Best Practice: Explicitly type the variable for clarity
        super().__init__()

        # 🧠 ML Signal: Iterating over a dictionary to access its items
        self._widget: ChartWidget = widget
        self._manager: BarManager = manager
        # 🧠 ML Signal: Using a method to get the bounding rectangle of a scene
        self._plots: dict[str, pg.GraphicsObject] = plots
        self._item_plot_map: dict[ChartItem, pg.GraphicsObject] = item_plot_map
        # 🧠 ML Signal: Checking if a point is within a rectangle

        self._x: int = 0
        # 🧠 ML Signal: Mapping a scene position to a view position
        self._y: float = 0
        self._plot_name: str = ""
        # ✅ Best Practice: Convert float to int explicitly

        self._init_ui()
        # 🧠 ML Signal: Accessing x and y coordinates of a point
        self._connect_signal()

    # 🧠 ML Signal: Iterating over dictionary values, common pattern for ML feature extraction

    # 🧠 ML Signal: Storing the plot name where the mouse is located
    def _init_ui(self) -> None:
        # 🧠 ML Signal: Method call on object, useful for dynamic behavior analysis
        """"""
        self._init_line()
        # 🧠 ML Signal: Calling update methods after processing input
        # 🧠 ML Signal: Method call on object, useful for dynamic behavior analysis
        self._init_label()
        self._init_info()

    # 🧠 ML Signal: Iterating over dictionary items, common pattern for ML feature extraction

    def _init_line(self) -> None:
        """
        Create line objects.
        # 🧠 ML Signal: Method call on object, useful for dynamic behavior analysis
        """
        # ✅ Best Practice: Type hinting improves code readability and maintainability.
        self._v_lines: dict[str, pg.InfiniteLine] = {}
        # 🧠 ML Signal: Method call on object, useful for dynamic behavior analysis
        self._h_lines: dict[str, pg.InfiniteLine] = {}
        # 🧠 ML Signal: Usage of external library method to get axis width.
        self._views: dict[str, pg.ViewBox] = {}

        # 🧠 ML Signal: Method call on object, useful for dynamic behavior analysis
        # 🧠 ML Signal: Usage of external library method to get axis height.
        pen: QtGui.QPen = pg.mkPen(WHITE_COLOR)
        # ✅ Best Practice: Type hinting improves code readability and maintainability.

        for plot_name, plot in self._plots.items():
            v_line: pg.InfiniteLine = pg.InfiniteLine(angle=90, movable=False, pen=pen)
            # ✅ Best Practice: Type hinting improves code readability and maintainability.
            h_line: pg.InfiniteLine = pg.InfiniteLine(angle=0, movable=False, pen=pen)
            view: pg.ViewBox = plot.getViewBox()
            # 🧠 ML Signal: Mapping scene coordinates to view coordinates.

            for line in [v_line, h_line]:
                line.setZValue(0)
                line.hide()
                view.addItem(line)

            # 🧠 ML Signal: Dynamic label text setting based on condition.
            self._v_lines[plot_name] = v_line
            self._h_lines[plot_name] = h_line
            self._views[plot_name] = view

    # 🧠 ML Signal: Setting position of a label in a plot.

    def _init_label(self) -> None:
        """
        Create label objects on axis.
        # ✅ Best Practice: Type hinting improves code readability and maintainability.
        """
        self._y_labels: dict[str, pg.TextItem] = {}
        # 🧠 ML Signal: Iterating over a mapping of items to plots
        for plot_name, plot in self._plots.items():
            # 🧠 ML Signal: Formatting datetime for display.
            label: pg.TextItem = pg.TextItem(
                plot_name, fill=CURSOR_COLOR, color=BLACK_COLOR
            )
            label.hide()
            # 🧠 ML Signal: Setting position of a label in a plot.
            label.setZValue(2)
            label.setFont(NORMAL_FONT)
            # 🧠 ML Signal: Setting anchor point for label positioning.
            plot.addItem(label, ignoreBounds=True)
            self._y_labels[plot_name] = label
        # 🧠 ML Signal: Iterating over plots to update information

        self._x_label: pg.TextItem = pg.TextItem(
            "datetime", fill=CURSOR_COLOR, color=BLACK_COLOR
        )
        # ⚠️ SAST Risk (Low): Potential KeyError if plot_name is not in self._infos
        self._x_label.hide()
        self._x_label.setZValue(2)
        self._x_label.setFont(NORMAL_FONT)
        plot.addItem(self._x_label, ignoreBounds=True)

    # ⚠️ SAST Risk (Low): Potential KeyError if plot_name is not in self._views

    def _init_info(self) -> None:
        """
        # ✅ Best Practice: Check boundary condition to prevent index out of range
        """
        self._infos: dict[str, pg.TextItem] = {}
        for plot_name, plot in self._plots.items():
            # 🧠 ML Signal: Incrementing a counter or index is a common pattern
            info: pg.TextItem = pg.TextItem(
                "info",
                # 🧠 ML Signal: Method call after state change indicates dependency on updated state
                color=CURSOR_COLOR,
                border=CURSOR_COLOR,
                fill=BLACK_COLOR,
                # ✅ Best Practice: Early return to handle edge case when cursor is at the start
            )
            info.hide()
            info.setZValue(2)
            # 🧠 ML Signal: Decrement operation on a variable, common in cursor or index manipulation
            info.setFont(NORMAL_FONT)
            plot.addItem(info)  # , ignoreBounds=True)
            # 🧠 ML Signal: Method call after state change, indicating a pattern of updating state
            self._infos[plot_name] = info

    def _connect_signal(self) -> None:
        """
        Connect mouse move signal to update function.
        """
        self._widget.scene().sigMouseMoved.connect(self._mouse_moved)

    # 🧠 ML Signal: Usage of object attributes to store state.

    def _mouse_moved(self, evt: tuple) -> None:
        """
        Callback function when mouse is moved.
        """
        if not self._manager.get_count():
            # ✅ Best Practice: Initialize variables to default values to ensure a clean state.
            return

        # First get current mouse point
        pos: tuple = evt
        # ✅ Best Practice: Use list() to create a copy of the dictionary values to avoid runtime errors if the dictionary is modified during iteration.

        for plot_name, view in self._views.items():
            # 🧠 ML Signal: Iterating over and hiding UI elements indicates a pattern of resetting or clearing a visual state.
            # ✅ Best Practice: Use list() to create a copy of the dictionary values to avoid runtime errors if the dictionary is modified during iteration.
            rect = view.sceneBoundingRect()

            if rect.contains(pos):
                mouse_point = view.mapSceneToView(pos)
                self._x = to_int(mouse_point.x())
                self._y = mouse_point.y()
                self._plot_name = plot_name
                break

        # Then update cursor component
        self._update_line()
        self._update_label()
        self.update_info()

    def _update_line(self) -> None:
        """"""
        for v_line in self._v_lines.values():
            v_line.setPos(self._x)
            v_line.show()

        for plot_name, h_line in self._h_lines.items():
            if plot_name == self._plot_name:
                h_line.setPos(self._y)
                h_line.show()
            else:
                h_line.hide()

    def _update_label(self) -> None:
        """"""
        bottom_plot: pg.PlotItem = list(self._plots.values())[-1]
        axis_width = bottom_plot.getAxis("right").width()
        axis_height = bottom_plot.getAxis("bottom").height()
        axis_offset: QtCore.QPointF = QtCore.QPointF(axis_width, axis_height)

        bottom_view: pg.ViewBox = list(self._views.values())[-1]
        bottom_right = bottom_view.mapSceneToView(
            bottom_view.sceneBoundingRect().bottomRight() - axis_offset
        )

        for plot_name, label in self._y_labels.items():
            if plot_name == self._plot_name:
                label.setText(str(self._y))
                label.show()
                label.setPos(bottom_right.x(), self._y)
            else:
                label.hide()

        dt: datetime | None = self._manager.get_datetime(self._x)
        if dt:
            self._x_label.setText(dt.strftime("%Y-%m-%d %H:%M:%S"))
            self._x_label.show()
            self._x_label.setPos(self._x, bottom_right.y())
            self._x_label.setAnchor((0, 0))

    def update_info(self) -> None:
        """"""
        buf: dict = {}

        for item, plot in self._item_plot_map.items():
            item_info_text: str = item.get_info_text(self._x)

            if plot not in buf:
                buf[plot] = item_info_text
            else:
                if item_info_text:
                    buf[plot] += "\n\n" + item_info_text

        for plot_name, plot in self._plots.items():
            plot_info_text: str = buf[plot]
            info: pg.TextItem = self._infos[plot_name]
            info.setText(plot_info_text)
            info.show()

            view: pg.ViewBox = self._views[plot_name]
            top_left = view.mapSceneToView(view.sceneBoundingRect().topLeft())
            info.setPos(top_left)

    def move_right(self) -> None:
        """
        Move cursor index to right by 1.
        """
        if self._x == self._manager.get_count() - 1:
            return
        self._x += 1

        self._update_after_move()

    def move_left(self) -> None:
        """
        Move cursor index to left by 1.
        """
        if self._x == 0:
            return
        self._x -= 1

        self._update_after_move()

    def _update_after_move(self) -> None:
        """
        Update cursor after moved by left/right.
        """
        bar: BarData | None = self._manager.get_bar(self._x)
        if bar is None:
            return

        self._y = bar.close_price

        self._update_line()
        self._update_label()

    def clear_all(self) -> None:
        """
        Clear all data.
        """
        self._x = 0
        self._y = 0
        self._plot_name = ""

        for line in list(self._v_lines.values()) + list(self._h_lines.values()):
            line.hide()

        for label in list(self._y_labels.values()) + [self._x_label]:
            label.hide()
