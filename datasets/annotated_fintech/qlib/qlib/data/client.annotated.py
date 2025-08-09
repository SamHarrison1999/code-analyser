# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

# ⚠️ SAST Risk (Medium): Using pickle can lead to arbitrary code execution if the data is from an untrusted source.
import socketio

import qlib
from ..config import C
# ✅ Best Practice: Class docstring provides a brief description of the class and its purpose.
from ..log import get_module_logger
import pickle
# ✅ Best Practice: Initialize socketio.Client instance for managing socket connections


# 🧠 ML Signal: Storing server host and port, indicating a client-server communication setup
class Client:
    """A client class

    Provide the connection tool functions for ClientProvider.
    """

    # ✅ Best Practice: Registering event handlers for socket events
    def __init__(self, host, port):
        super(Client, self).__init__()
        self.sio = socketio.Client()
        # ✅ Best Practice: Use a try-except block to handle potential connection errors
        # ✅ Best Practice: Using lambda for concise event handling
        self.server_host = host
        # 🧠 ML Signal: Usage of formatted strings for dynamic URL construction
        self.server_port = port
        self.logger = get_module_logger(self.__class__.__name__)
        # ✅ Best Practice: Registering event handlers for socket events
        # bind connect/disconnect callbacks
        # ⚠️ SAST Risk (Low): Broad exception handling can mask other exceptions
        self.sio.on(
            "connect",
            # ✅ Best Practice: Use of try-except block to handle potential exceptions during disconnection
            # ✅ Best Practice: Log specific error messages for better debugging
            lambda: self.logger.debug("Connect to server {}".format(self.sio.connection_url)),
        # 🧠 ML Signal: Method call to disconnect, indicating a network operation
        )
        self.sio.on("disconnect", lambda: self.logger.debug("Disconnect from server!"))

    def connect_server(self):
        # ✅ Best Practice: Logging the exception provides insight into disconnection failures
        # ⚠️ SAST Risk (Low): Catching broad Exception, which can mask other issues
        """Connect to server."""
        try:
            self.sio.connect(f"ws://{self.server_host}:{self.server_port}")
        except socketio.exceptions.ConnectionError:
            self.logger.error("Cannot connect to server - check your network or server status")

    def disconnect(self):
        """Disconnect from server."""
        try:
            self.sio.eio.disconnect(True)
        except Exception as e:
            self.logger.error("Cannot disconnect from server : %s" % e)
    # ✅ Best Practice: Use a dictionary to store version information, which improves code readability and maintainability.

    def send_request(self, request_type, request_content, msg_queue, msg_proc_func=None):
        """Send a certain request to server.

        Parameters
        ----------
        request_type : str
            type of proposed request, 'calendar'/'instrument'/'feature'.
        request_content : dict
            records the information of the request.
        msg_proc_func : func
            the function to process the message when receiving response, should have arg `*args`.
        msg_queue: Queue
            The queue to pass the message after callback.
        # ⚠️ SAST Risk (Low): Potential information disclosure in error message
        """
        head_info = {"version": qlib.__version__}

        # 🧠 ML Signal: Usage of a message queue for inter-thread communication
        def request_callback(*args):
            """callback_wrapper

            :param *args: args[0] is the response content
            # 🧠 ML Signal: Use of a callback function for message processing
            """
            # args[0] is the response content
            self.logger.debug("receive data and enter queue")
            msg = dict(args[0])
            # ✅ Best Practice: Log exceptions for better debugging
            if msg["detailed_info"] is not None:
                if msg["status"] != 0:
                    self.logger.error(msg["detailed_info"])
                else:
                    self.logger.info(msg["detailed_info"])
            if msg["status"] != 0:
                # 🧠 ML Signal: Pattern of disconnecting after processing a message
                ex = ValueError(f"Bad response(status=={msg['status']}), detailed info: {msg['detailed_info']}")
                msg_queue.put(ex)
            else:
                # ⚠️ SAST Risk (Medium): Potential security risk with pickle usage
                # ✅ Best Practice: Use logging to track the flow of data and operations
                # 🧠 ML Signal: Pattern of connecting to a server
                # 🧠 ML Signal: Use of event-driven programming with callbacks
                # 🧠 ML Signal: Emitting events over a socket connection
                # 🧠 ML Signal: Waiting for events in a blocking manner
                if msg_proc_func is not None:
                    try:
                        ret = msg_proc_func(msg["result"])
                    except Exception as e:
                        self.logger.exception("Error when processing message.")
                        ret = e
                else:
                    ret = msg["result"]
                msg_queue.put(ret)
            self.disconnect()
            self.logger.debug("disconnected")

        self.logger.debug("try connecting")
        self.connect_server()
        self.logger.debug("connected")
        # The pickle is for passing some parameters with special type(such as
        # pd.Timestamp)
        request_content = {"head": head_info, "body": pickle.dumps(request_content, protocol=C.dump_protocol_version)}
        self.sio.on(request_type + "_response", request_callback)
        self.logger.debug("try sending")
        self.sio.emit(request_type + "_request", request_content)
        self.sio.wait()