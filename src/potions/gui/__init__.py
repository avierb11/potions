"""
Interactive Streamlit GUIs for `potions`.

The package currently provides one GUI: the East River lateral-model
calibration app (`potions.gui.app`).  It is an interactive
calibration workbench for the East River three-layer lateral
(snow / surface / ground) hydrology–reactive-transport model: a
sidebar slider for every calibrated parameter, the hydrology and
reactive-transport models re-run on every change, a per-run
objective-function table, and figures of streamflow and river
concentrations, zone concentrations and storage, and mineral
reaction rates.

Start the GUI with the console script installed with the package::

    potions-gui

or programmatically with `main`.  Arguments are forwarded to
streamlit's ``run`` command, so server flags work as expected::

    potions-gui --server.port 8502
"""

from collections.abc import Sequence

__all__ = ["main"]


# ==== Functions ==== #
def main(argv: Sequence[str] | None = None) -> None:
    """
    Launch the East River lateral-model calibration GUI.

    Runs the GUI's entry script (`potions.gui.app`) through
    streamlit's own ``run`` command in the current process, so the
    app that is served is the installed package copy of the script
    and the server runs in the foreground until interrupted.

    Args:
        argv: Argument list forwarded to ``streamlit run`` after the
            app script path (e.g. ``["--server.port", "8502"]``).
            When omitted, ``sys.argv[1:]`` is forwarded, so flags
            typed after the console script name pass through
            unchanged.
    """
    import sys
    from pathlib import Path

    import streamlit.web.cli as st_cli

    app_script: Path = Path(__file__).resolve().with_name("app.py")
    forwarded: list[str]
    if argv is None:
        forwarded = list(sys.argv[1:])
    else:
        forwarded = list(argv)
    st_cli.main(["run", str(app_script), *forwarded])
# =================== #
