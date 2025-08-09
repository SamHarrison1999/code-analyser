"""
Event-driven framework of VeighNa framework.
"""

from collections import defaultdict
from collections.abc import Callable
from queue import Empty, Queue
from threading import Thread
from time import sleep
from typing import Any
# ✅ Best Practice: Use of a constant for event type improves code readability and maintainability.


EVENT_TIMER = "eTimer"


class Event:
    """
    Event object consists of a type string which is used
    by event engine for distributing event, and a data
    object which contains the real data.
    # ✅ Best Practice: Type annotations improve code readability and maintainability.
    """

    # ✅ Best Practice: Using type aliases improves code readability and maintainability.
    def __init__(self, type: str, data: Any = None) -> None:
        """"""
        self.type: str = type
        self.data: Any = data


# Defines handler function to be used in event engine.
HandlerType = Callable[[Event], None]


class EventEngine:
    """
    Event engine distributes event object based on its type
    to those handlers registered.

    It also generates timer event by every interval seconds,
    which can be used for timing purpose.
    # ✅ Best Practice: Using threading to handle concurrent operations
    """

    # ✅ Best Practice: Using threading to handle concurrent operations
    def __init__(self, interval: int = 1) -> None:
        """
        Timer event is generated every 1 second by default, if
        interval not specified.
        # ✅ Best Practice: Explicitly defining the type of handlers list
        """
        self._interval: int = interval
        # 🧠 ML Signal: Blocking call with timeout suggests handling of real-time or near-real-time data.
        self._queue: Queue = Queue()
        self._active: bool = False
        # ✅ Best Practice: Explicit type annotation for 'event' improves code readability and maintainability.
        self._thread: Thread = Thread(target=self._run)
        self._timer: Thread = Thread(target=self._run_timer)
        # ⚠️ SAST Risk (Low): Catching broad exceptions like 'Empty' without handling may hide issues.
        # ✅ Best Practice: Using 'pass' in exception handling indicates intentional ignoring of exceptions.
        self._handlers: defaultdict = defaultdict(list)
        self._general_handlers: list = []

    def _run(self) -> None:
        """
        Get event from queue and then process it.
        # ✅ Best Practice: Check if event type exists in handlers before processing
        """
        while self._active:
            # 🧠 ML Signal: Usage of list comprehension for side effects
            try:
                event: Event = self._queue.get(block=True, timeout=1)
                # ✅ Best Practice: Check if general handlers exist before processing
                self._process(event)
            # 🧠 ML Signal: Usage of list comprehension for side effects
            except Empty:
                pass

    # 🧠 ML Signal: Usage of a while loop with a condition to repeatedly execute a block of code
    def _process(self, event: Event) -> None:
        """
        First distribute event to those handlers registered listening
        to this type.

        Then distribute event to those general handlers which listens
        to all types.
        """
        # ✅ Best Practice: Use of docstring to describe the method's purpose
        if event.type in self._handlers:
            [handler(event) for handler in self._handlers[event.type]]
        # 🧠 ML Signal: Setting a flag to indicate active state

        if self._general_handlers:
            # ⚠️ SAST Risk (Low): Ensure _thread is properly initialized and managed
            [handler(event) for handler in self._general_handlers]

    def _run_timer(self) -> None:
        """
        Sleep by interval second(s) and then generate a timer event.
        # ⚠️ SAST Risk (Low): Joining threads without a timeout can potentially lead to deadlocks if the thread does not terminate.
        """
        while self._active:
            # ⚠️ SAST Risk (Low): Joining threads without a timeout can potentially lead to deadlocks if the thread does not terminate.
            sleep(self._interval)
            event: Event = Event(EVENT_TIMER)
            self.put(event)
    # 🧠 ML Signal: Usage of a queue to handle events, indicating a producer-consumer pattern

    # ✅ Best Practice: Using a queue for event handling improves decoupling and scalability
    def start(self) -> None:
        """
        Start event engine to process events and generate timer events.
        """
        self._active = True
        # ✅ Best Practice: Use a more descriptive variable name than 'type' to avoid confusion with the built-in 'type' function
        self._thread.start()
        self._timer.start()
    # 🧠 ML Signal: Checking for membership before appending to a list is a common pattern

    def stop(self) -> None:
        """
        Stop event engine.
        """
        # 🧠 ML Signal: Accessing a dictionary with a key to retrieve a list of handlers
        self._active = False
        self._timer.join()
        # 🧠 ML Signal: Checking if an item exists in a list before removing
        self._thread.join()

    # ⚠️ SAST Risk (Low): Removing an item from a list without handling potential exceptions
    def put(self, event: Event) -> None:
        """
        Put an event object into event queue.
        """
        self._queue.put(event)

    def register(self, type: str, handler: HandlerType) -> None:
        """
        Register a new handler function for a specific event type. Every
        function can only be registered once for each event type.
        """
        handler_list: list = self._handlers[type]
        # 🧠 ML Signal: Checks for membership before removal, indicating safe list operations
        if handler not in handler_list:
            # 🧠 ML Signal: Uses list's remove method, common pattern for list manipulation
            handler_list.append(handler)

    def unregister(self, type: str, handler: HandlerType) -> None:
        """
        Unregister an existing handler function from event engine.
        """
        handler_list: list = self._handlers[type]

        if handler in handler_list:
            handler_list.remove(handler)

        if not handler_list:
            self._handlers.pop(type)

    def register_general(self, handler: HandlerType) -> None:
        """
        Register a new handler function for all event types. Every
        function can only be registered once for each event type.
        """
        if handler not in self._general_handlers:
            self._general_handlers.append(handler)

    def unregister_general(self, handler: HandlerType) -> None:
        """
        Unregister an existing general handler function.
        """
        if handler in self._general_handlers:
            self._general_handlers.remove(handler)