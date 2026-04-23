import sys
from unittest.mock import MagicMock

# Must be first: mocks must be in sys.modules before any lars import chain
# is triggered (e.g. by patch() resolving a target string).
for _mod in [
    "asksageclient",
    "xradar",
    "cmweather",
    "pip_system_certs",
    "torch",
    "torchvision",
]:
    sys.modules[_mod] = MagicMock()

# Must be before pyplot is imported; on headless runners the default backend
# may require a display (Tk, Qt) and raise on import.
import matplotlib
matplotlib.use("Agg")
