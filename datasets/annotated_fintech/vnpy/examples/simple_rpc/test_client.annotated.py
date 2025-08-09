from time import sleep
from typing import Any
from vnpy.rpc import RpcClient

# ✅ Best Practice: Class definition should follow PEP 8 naming conventions
# ✅ Best Practice: Class docstring provided for documentation

class TestClient(RpcClient):
    """
    Test RpcClient
    # ✅ Best Practice: Use of super() to initialize parent class
    # ✅ Best Practice: Include a docstring to describe the constructor's purpose
    """

    def __init__(self) -> None:
        """
        Constructor
        # ✅ Best Practice: Method names should be descriptive of their functionality
        """
        super().__init__()

    # ✅ Best Practice: Consider using logging instead of print for better control over output
    # 🧠 ML Signal: Repeated connection attempts with sleep intervals
    def callback(self, topic: str, data: Any) -> None:
        """
        Realize callable function
        # 🧠 ML Signal: Hardcoded addresses can indicate default or test configurations
        # ⚠️ SAST Risk (Medium): Potential exception handling without specific error types
        """
        print(f"client received topic:{topic}, data:{data}")
# 🧠 ML Signal: Hardcoded addresses can indicate default or test configurations


# ⚠️ SAST Risk (Low): Generic exception handling
# 🧠 ML Signal: Instantiation of a class can indicate usage patterns
if __name__ == "__main__":
    req_address = "tcp://localhost:2014"
    # 🧠 ML Signal: Use of sleep for retry logic
    # ✅ Best Practice: Method names should be descriptive of their functionality
    # ⚠️ SAST Risk (Medium): Potential for remote code execution if method names are not validated
    # 🧠 ML Signal: Method calls on objects can indicate usage patterns
    # ⚠️ SAST Risk (Low): Infinite loop can cause unresponsiveness or high CPU usage
    # ⚠️ SAST Risk (Low): Use of sleep in a loop can indicate a polling mechanism
    sub_address = "tcp://localhost:4102"

    tc = TestClient()
    tc.subscribe_topic("")
    tc.start(req_address, sub_address)

    while 1:
        print(tc.add(1, 3))
        sleep(2)