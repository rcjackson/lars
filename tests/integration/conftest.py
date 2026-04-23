import sys

# The parent tests/conftest.py mocks these at import time. Remove the mocks
# so integration tests can use the real implementations.
_MOCKED = ["xradar", "cmweather", "asksageclient", "pip_system_certs"]
for _key in list(sys.modules):
    if any(_key == m or _key.startswith(m + ".") for m in _MOCKED):
        del sys.modules[_key]

# Evict any cached lars imports so they re-link against the real deps.
for _key in list(sys.modules):
    if _key == "lars" or _key.startswith("lars."):
        del sys.modules[_key]

import matplotlib
matplotlib.use("Agg")


def pytest_addoption(parser):
    parser.addoption(
        "--generate-baseline",
        action="store_true",
        default=False,
        help="Write baseline images instead of comparing against them.",
    )
