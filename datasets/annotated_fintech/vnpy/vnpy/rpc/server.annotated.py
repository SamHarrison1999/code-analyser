import threading
import traceback
from time import time
# ✅ Best Practice: Use of collections.abc.Callable for type hinting improves code readability and maintainability.
from collections.abc import Callable

import zmq
# 🧠 ML Signal: Importing specific constants from a module indicates usage patterns and dependencies.

from .common import HEARTBEAT_TOPIC, HEARTBEAT_INTERVAL
# ✅ Best Practice: Consider adding a class docstring to describe the purpose and usage of the RpcServer class.


class RpcServer:
    """"""
    # ✅ Best Practice: Use of type annotations for better code readability and maintainability

    def __init__(self) -> None:
        """
        Constructor
        # ✅ Best Practice: Use of type annotations for better code readability and maintainability
        """
        # Save functions dict: key is function name, value is function object
        # ✅ Best Practice: Use of type annotations for better code readability and maintainability
        self._functions: dict[str, Callable] = {}

        # ✅ Best Practice: Use of type annotations for better code readability and maintainability
        # ✅ Best Practice: Method should have a descriptive docstring explaining its purpose
        # Zmq port related
        self._context: zmq.Context = zmq.Context()
        # ✅ Best Practice: Use of type annotations for better code readability and maintainability
        # 🧠 ML Signal: Method returning a boolean value indicating state

        # ✅ Best Practice: Use of type annotations for better code readability and maintainability
        # Reply socket (Request–reply pattern)
        self._socket_rep: zmq.Socket = self._context.socket(zmq.REP)

        # ✅ Best Practice: Use of type annotations for better code readability and maintainability
        # Publish socket (Publish–subscribe pattern)
        self._socket_pub: zmq.Socket = self._context.socket(zmq.PUB)

        # Worker thread related
        self._active: bool = False                          # RpcServer status
        # ✅ Best Practice: Check if the server is already active to prevent redundant operations
        self._thread: threading.Thread | None = None        # RpcServer thread
        self._lock: threading.Lock = threading.Lock()

        # ⚠️ SAST Risk (Medium): Binding to addresses without validation can lead to security vulnerabilities
        # Heartbeat related
        self._heartbeat_at: float | None = None
    # ⚠️ SAST Risk (Medium): Binding to addresses without validation can lead to security vulnerabilities

    def is_active(self) -> bool:
        """"""
        # ✅ Best Practice: Using a separate thread to run the server allows for non-blocking operations
        return self._active
    # 🧠 ML Signal: Starting a thread is a common pattern for asynchronous operations

    def start(
        self,
        # 🧠 ML Signal: Setting a heartbeat interval is a common pattern for maintaining server health
        # ✅ Best Practice: Check if the server is active before attempting to stop it
        rep_address: str,
        pub_address: str,
    ) -> None:
        """
        Start RpcServer
        # ✅ Best Practice: Check if the thread is alive before joining to avoid unnecessary blocking.
        """
        if self._active:
            # ✅ Best Practice: Resetting the thread reference to None after joining to prevent reuse.
            return

        # Bind socket address
        self._socket_rep.bind(rep_address)
        self._socket_pub.bind(pub_address)

        # ✅ Best Practice: Regularly checking for a heartbeat ensures the server is responsive and can detect failures.
        # Start RpcServer status
        self._active = True

        # Start RpcServer thread
        # 🧠 ML Signal: Receiving and processing objects from a socket can indicate a pattern of network communication.
        self._thread = threading.Thread(target=self.run)
        self._thread.start()
        # 🧠 ML Signal: Unpacking received data into specific variables can indicate expected data structure.

        # Init heartbeat publish timestamp
        # ⚠️ SAST Risk (Medium): Accessing a function by name from a dictionary can lead to code execution risks if not properly validated.
        self._heartbeat_at = time() + HEARTBEAT_INTERVAL

    def stop(self) -> None:
        """
        Stop RpcServer
        """
        if not self._active:
            # ✅ Best Practice: Catching exceptions and formatting traceback helps in debugging and error reporting.
            return

        # Stop RpcServer status
        # 🧠 ML Signal: Sending objects over a socket can indicate a pattern of network communication.
        # ✅ Best Practice: Use of a lock to ensure thread safety when accessing shared resources
        self._active = False
    # ✅ Best Practice: Closing sockets when done to release resources and avoid potential leaks.

    # ✅ Best Practice: Add type hint for the return type of the function
    # 🧠 ML Signal: Usage of socket communication for data transmission
    def join(self) -> None:
        # ⚠️ SAST Risk (Low): Potential data leakage if sensitive information is sent over an insecure channel
        # Wait for RpcServer thread to exit
        if self._thread and self._thread.is_alive():
            self._thread.join()
        self._thread = None
    # ✅ Best Practice: Include type hints for function return type
    # 🧠 ML Signal: Usage of function registration pattern

    # ⚠️ SAST Risk (Low): Potential risk of overwriting existing functions with the same name
    def run(self) -> None:
        """
        Run RpcServer functions
        """
        # ✅ Best Practice: Use descriptive variable names
        while self._active:
            # Poll response socket for 1 second
            # ⚠️ SAST Risk (Low): Potential time-based logic flaw if time is manipulated
            # 🧠 ML Signal: Usage of publish-subscribe pattern
            # 🧠 ML Signal: Pattern of updating state with time-based logic
            n: int = self._socket_rep.poll(1000)
            self.check_heartbeat()

            if not n:
                continue

            # Receive request data from Reply socket
            req = self._socket_rep.recv_pyobj()

            # Get function name and parameters
            name, args, kwargs = req

            # Try to get and execute callable function object; capture exception information if it fails
            try:
                func: Callable = self._functions[name]
                r: object = func(*args, **kwargs)
                rep: list = [True, r]
            except Exception as e:  # noqa
                rep = [False, traceback.format_exc()]

            # send callable response by Reply socket
            self._socket_rep.send_pyobj(rep)

        # Unbind socket address
        self._socket_pub.close()
        self._socket_rep.close()

    def publish(self, topic: str, data: object) -> None:
        """
        Publish data
        """
        with self._lock:
            self._socket_pub.send_pyobj([topic, data])

    def register(self, func: Callable) -> None:
        """
        Register function
        """
        self._functions[func.__name__] = func

    def check_heartbeat(self) -> None:
        """
        Check whether it is required to send heartbeat.
        """
        now: float = time()

        if self._heartbeat_at and now >= self._heartbeat_at:
            # Publish heartbeat
            self.publish(HEARTBEAT_TOPIC, now)

            # Update timestamp of next publish
            self._heartbeat_at = now + HEARTBEAT_INTERVAL