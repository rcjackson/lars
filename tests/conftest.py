import sys
import matplotlib
import matplotlib.pyplot as plt
from unittest.mock import MagicMock

matplotlib.use("Agg")

# Mock packages not available in CI (private or heavy scientific deps)
for _mod in [
    "asksageclient",
    "xradar",
    "cmweather",
    "pip_system_certs",
    "torch",
    "torchvision",
]:
    sys.modules[_mod] = MagicMock()

plt.close("all")
