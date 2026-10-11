"""Test-wide setup: mocks for everything, real services only on request.

Every test runs against the mock providers, whatever .env says. config.py
loads .env without overriding variables already in the environment, so
setting them here — before any test module imports config — wins. Without
this, PROVIDER_RIGGING=real in a developer's .env sent ~27 ordinary tests to
a live seg_server, and the suite's verdict depended on whether it was up.

Tests marked ``real`` exercise the real animation path against the
seg_server in .env (RIGGING_SERVICE_URL). They are skipped unless pytest is
run with ``--real``, and skip with a reason if the service can't be reached.
"""

from __future__ import annotations

import os

import pytest

for _name in ("PROVIDER_RIGGING", "PROVIDER_POSING", "PROVIDER_TRAINING"):
    os.environ[_name] = "mock"


def pytest_addoption(parser):
    parser.addoption(
        "--real", action="store_true", default=False,
        help="also run tests marked 'real', against the seg_server in .env")


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "real: needs the live seg_server in .env; runs only with --real")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--real"):
        return
    skip = pytest.mark.skip(reason="needs the live seg_server; run with --real")
    for item in items:
        if "real" in item.keywords:
            item.add_marker(skip)
