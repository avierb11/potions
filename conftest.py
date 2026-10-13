"""Test-suite configuration.

Disables the on-by-default run-result saving for the whole test suite: the
other tests make dozens of hydrologic model runs that would each create a
`results/run_<timestamp>/` folder in the current working directory. The
saving behavior itself is covered by
``src/potions/tests/test_result_saving.py``, where tests opt in explicitly
(usually writing under pytest's ``tmp_path``).
"""

import os


def pytest_configure(config) -> None:
    os.environ["POTIONS_SAVE_RESULTS"] = "0"
