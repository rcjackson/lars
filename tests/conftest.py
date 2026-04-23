import sys
from unittest.mock import MagicMock

# Mock packages not available in CI (private or heavy scientific deps)
for _mod in [
    "asksageclient",
    "xradar",
    "cmweather",
    "pip_system_certs",
    "sklearn",
    "sklearn.metrics",
    "sklearn.preprocessing",
    "torch",
    "torchvision",
]:
    sys.modules[_mod] = MagicMock()
