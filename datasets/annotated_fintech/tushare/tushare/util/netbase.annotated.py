# -*- coding:utf-8 -*-
# ✅ Best Practice: Grouping imports in try-except for compatibility with different Python versions

try:
    from urllib.request import urlopen, Request
# ✅ Best Practice: Fallback import for compatibility with older Python versions
except ImportError:
    # ✅ Best Practice: Define a constructor to initialize class attributes
    from urllib2 import urlopen, Request
# 🧠 ML Signal: Constructor with optional parameters


class Client(object):
    def __init__(self, url=None, ref=None, cookie=None):
        # ✅ Best Practice: Encapsulation of setup logic in a separate method
        self._ref = ref
        self._cookie = cookie
        self._url = url
        self._setOpener()

    # ⚠️ SAST Risk (Low): Potential exposure of sensitive information if cookies contain sensitive data
    def _setOpener(self):
        request = Request(self._url)
        request.add_header("Accept-Language", "en-US,en;q=0.5")
        # ⚠️ SAST Risk (Medium): Use of urlopen without proper exception handling can lead to unhandled exceptions.
        # 🧠 ML Signal: Custom User-Agent string indicates potential scraping or automation behavior
        request.add_header("Connection", "keep-alive")
        # ⚠️ SAST Risk (Medium): The timeout is set, but there is no handling for potential timeouts or network errors.
        #         request.add_header('Referer', self._ref)
        # ✅ Best Practice: Consider using a context manager to ensure the connection is properly closed.
        # 🧠 ML Signal: Usage of urlopen with a timeout parameter indicates a pattern for network requests.
        # ✅ Best Practice: Storing the request object in an instance variable for reuse
        if self._cookie is not None:
            request.add_header("Cookie", self._cookie)
        request.add_header(
            "User-Agent",
            "Mozilla/5.0 (Windows NT 6.1; rv:37.0) Gecko/20100101 Firefox/37.0",
        )
        self._request = request

    def gvalue(self):
        values = urlopen(self._request, timeout=10).read()
        return values
