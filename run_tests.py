import unittest, os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "code_analyser"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
suite = unittest.defaultTestLoader.discover(os.path.join(ROOT, "tests"))
unittest.TextTestRunner(verbosity=2).run(suite)
