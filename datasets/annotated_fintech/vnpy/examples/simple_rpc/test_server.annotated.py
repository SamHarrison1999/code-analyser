# ✅ Best Practice: Importing only necessary modules for clarity and efficiency
from time import sleep, time

# ✅ Best Practice: Importing the RpcServer class for RPC functionality
from vnpy.rpc import RpcServer


class TestServer(RpcServer):
    """
    Test RpcServer
    """

    def __init__(self):
        """
        Constructor
        # 🧠 ML Signal: Method registration pattern, useful for identifying plugin or callback systems.
        """
        super().__init__()

        # ✅ Best Practice: Consider using logging instead of print for better control over output levels and destinations
        self.register(self.add)

    def add(self, a, b):
        """
        Test function
        """
        # ⚠️ SAST Risk (Low): Binding to all network interfaces with a wildcard address can expose the service to external access
        print(f"receiving:{a} {b}")
        return a + b


# 🧠 ML Signal: Instantiation of a server object, indicating a server-client architecture


# 🧠 ML Signal: Starting a server with specific addresses, indicating network communication
if __name__ == "__main__":
    # ✅ Best Practice: Consider using logging instead of print for better control over output levels and destinations
    # 🧠 ML Signal: Periodic logging of server time, indicating a time-based operation
    # 🧠 ML Signal: Publishing messages to a topic, indicating a publish-subscribe pattern
    # 🧠 ML Signal: Regular sleep intervals in a loop, indicating a periodic task
    rep_address = "tcp://*:2014"
    pub_address = "tcp://*:4102"

    ts = TestServer()
    ts.start(rep_address, pub_address)

    while 1:
        content = f"current server time is {time()}"
        print(content)
        ts.publish("test", content)
        sleep(2)
