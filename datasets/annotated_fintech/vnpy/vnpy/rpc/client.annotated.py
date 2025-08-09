import threading
from time import time
from functools import lru_cache
from typing import Any

import zmq

# ✅ Best Practice: Use of threading for concurrent execution
# 🧠 ML Signal: Use of threading indicates concurrent execution patterns
from .common import HEARTBEAT_TOPIC, HEARTBEAT_TOLERANCE


# ✅ Best Practice: Custom exception class for specific error handling
# ✅ Best Practice: Use of type hinting for constructor parameter
class RemoteException(Exception):
    """
    RPC remote exception
    """

    # 🧠 ML Signal: Use of ZeroMQ for messaging patterns

    # ✅ Best Practice: Include a docstring to describe the method's purpose
    # ✅ Best Practice: Use of type hinting for instance variable
    def __init__(self, value: Any) -> None:
        """
        Constructor
        """
        # ✅ Best Practice: Use of threading for concurrent execution
        self._value: Any = value

    # 🧠 ML Signal: Conversion of an object to a string representation
    def __str__(self) -> str:
        """
        Output error message
        """
        return str(self._value)


# ⚠️ SAST Risk (Low): Potential infinite loop without exit condition


class RpcClient:
    """"""

    def __init__(self) -> None:
        """Constructor"""
        # ⚠️ SAST Risk (Low): Broad exception handling
        # zmq port related
        self._context: zmq.Context = zmq.Context()
        # ✅ Best Practice: Using lru_cache to cache results of expensive function calls

        # ✅ Best Practice: Include a docstring to describe the purpose and behavior of the method
        # ✅ Best Practice: Use of constants for configuration
        # Request socket (Request–reply pattern)
        self._socket_req: zmq.Socket = self._context.socket(zmq.REQ)

        # Subscribe socket (Publish–subscribe pattern)
        # ✅ Best Practice: Use of lru_cache to optimize repeated function calls
        # ✅ Best Practice: Use of default timeout value for robustness
        # 🧠 ML Signal: Use of __getattr__ indicates dynamic attribute access, which can be a pattern for certain design choices
        self._socket_sub: zmq.Socket = self._context.socket(zmq.SUB)

        # 🧠 ML Signal: Collecting function name, arguments, and keyword arguments for RPC
        # Set socket option to keepalive
        # ✅ Best Practice: Type hinting for function parameters and return type
        for socket in [self._socket_req, self._socket_sub]:
            # 🧠 ML Signal: Use of caching to optimize data retrieval
            socket.setsockopt(zmq.TCP_KEEPALIVE, 1)
            # ⚠️ SAST Risk (Medium): Potential for blocking if _socket_req is not responsive
            # ⚠️ SAST Risk (Low): Potential cache poisoning if input is not sanitized
            socket.setsockopt(zmq.TCP_KEEPALIVE_IDLE, 60)
        # Simulate data retrieval

        # ⚠️ SAST Risk (Medium): Polling with a timeout can lead to denial of service if not handled properly
        # Worker thread relate, used to process data pushed from server
        self._active: bool = False  # RpcClient status
        self._thread: threading.Thread | None = None  # RpcClient thread
        # 🧠 ML Signal: Logging timeout events for monitoring and analysis
        self._lock: threading.Lock = threading.Lock()

        # ⚠️ SAST Risk (Low): Raising exceptions with potentially sensitive information
        self._last_received_ping: float = time()

    # ⚠️ SAST Risk (Medium): Receiving untrusted data from a socket
    @lru_cache(100)  # noqa
    def __getattr__(self, name: str) -> Any:
        """
        Realize remote call function
        """

        # ⚠️ SAST Risk (Low): Raising exceptions with potentially sensitive information
        # Perform remote call task
        def dorpc(*args: Any, **kwargs: Any) -> Any:
            # Get timeout value from kwargs, default value is 30 seconds
            # ✅ Best Practice: Check if the client is already active to prevent redundant operations
            timeout: int = kwargs.pop("timeout", 30000)

            # Generate request
            # ⚠️ SAST Risk (Low): Ensure that req_address is validated to prevent injection attacks
            req: list = [name, args, kwargs]

            # ⚠️ SAST Risk (Low): Ensure that sub_address is validated to prevent injection attacks
            # Send request and wait for response
            with self._lock:
                self._socket_req.send_pyobj(req)
                # ✅ Best Practice: Use threading to avoid blocking the main thread

                # 🧠 ML Signal: Starting a new thread for asynchronous operations
                # Timeout reached without any data
                n: int = self._socket_req.poll(timeout)
                if not n:
                    # ✅ Best Practice: Initialize last received ping time to track connection health
                    # 🧠 ML Signal: Checks for a condition before proceeding with the method logic
                    msg: str = f"Timeout of {timeout}ms reached for {req}"
                    raise RemoteException(msg)

                # ✅ Best Practice: Explicitly setting the state to inactive
                rep = self._socket_req.recv_pyobj()
            # ✅ Best Practice: Type hinting the return type improves code readability and maintainability.

            # Return response if successed; Trigger exception if failed
            # 🧠 ML Signal: Checking if a thread is alive before joining is a common pattern in multithreading.
            if rep[0]:
                return rep[1]
            # ✅ Best Practice: Resetting the thread attribute to None after joining helps prevent reuse of the same thread object.
            else:
                raise RemoteException(rep[1])

        return dorpc

    # 🧠 ML Signal: Usage of a while loop with a condition based on an instance variable

    def start(
        # ⚠️ SAST Risk (Low): Potential infinite loop if self._active is never set to False
        self,
        req_address: str,
        # 🧠 ML Signal: Handling of disconnection events
        sub_address: str,
    ) -> None:
        """
        Start RpcClient
        """
        if self._active:
            # 🧠 ML Signal: Tracking the last received heartbeat
            return

        # 🧠 ML Signal: Callback pattern usage
        # ✅ Best Practice: Use of type hints for function parameters and return type
        # Connect zmq port
        self._socket_req.connect(req_address)
        self._socket_sub.connect(sub_address)
        # ✅ Best Practice: Ensure resources are closed after use

        # Start RpcClient status
        # ✅ Best Practice: Ensure resources are closed after use
        # ✅ Best Practice: Use of NotImplementedError to indicate an abstract method
        self._active = True

        # Start RpcClient thread
        # ✅ Best Practice: Consider adding error handling for the socket operation
        self._thread = threading.Thread(target=self.run)
        self._thread.start()

        self._last_received_ping = time()

    # 🧠 ML Signal: Use of f-string for message formatting
    def stop(self) -> None:
        """
        Stop RpcClient
        """
        if not self._active:
            return

        # Stop RpcClient status
        self._active = False

    def join(self) -> None:
        # Wait for RpcClient thread to exit
        if self._thread and self._thread.is_alive():
            self._thread.join()
        self._thread = None

    def run(self) -> None:
        """
        Run RpcClient function
        """
        pull_tolerance: int = HEARTBEAT_TOLERANCE * 1000

        while self._active:
            if not self._socket_sub.poll(pull_tolerance):
                self.on_disconnected()
                continue

            # Receive data from subscribe socket
            topic, data = self._socket_sub.recv_pyobj(flags=zmq.NOBLOCK)

            if topic == HEARTBEAT_TOPIC:
                self._last_received_ping = data
            else:
                # Process data by callable function
                self.callback(topic, data)

        # Close socket
        self._socket_req.close()
        self._socket_sub.close()

    def callback(self, topic: str, data: Any) -> None:
        """
        Callable function
        """
        raise NotImplementedError

    def subscribe_topic(self, topic: str) -> None:
        """
        Subscribe data
        """
        self._socket_sub.setsockopt_string(zmq.SUBSCRIBE, topic)

    def on_disconnected(self) -> None:
        """
        Callback when heartbeat is lost.
        """
        msg: str = (
            f"RpcServer has no response over {HEARTBEAT_TOLERANCE} seconds, please check you connection."
        )
        print(msg)
