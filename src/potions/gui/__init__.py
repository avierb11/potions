"""
Interactive GUI for `potions`.

The package provides one GUI: the East River lateral-model
calibration workbench (`potions.gui.window`).  It is a PySide6
desktop app: the model's three figures in a single panel (flipped
by arrow buttons or the Left / Right keys) over a row that holds
the per-run objective-function table on the left and every
calibrated parameter on the right, grouped by zone (hydrology in
blue, reactive transport in red).  Parameter changes re-run the models on
a background thread; a *Save run* click stores the run's model
output together with the session state, which the app restores on
its next start.

Start the GUI with the console script installed with the package::

    potions-gui

or programmatically with `main`.  The only option is ``--csv``::

    potions-gui --csv path/to/series.csv
"""

from collections.abc import Sequence
from pathlib import Path

__all__ = ["main"]


# ==== Functions ==== #
def main(argv: Sequence[str] | None = None) -> None:
    """
    Launch the East River lateral-model calibration GUI.

    Creates the `QApplication` (the Fusion style, for a consistent
    base), sets the publication rcParams, loads the calibration
    inputs, and shows the main window.  The window restores the
    most recent saved run at startup (or, with no saved run, starts
    from the default parameters and runs the models once in the
    background); the process ends when the window closes.

    Args:
        argv: Argument list to parse; when omitted,
            ``sys.argv[1:]`` is parsed, so flags typed after the
            console script name pass through unchanged.  The
            supported option is ``--csv``, the path of the series
            CSV to use (default: the record shipped with the
            package).
    """
    import argparse
    import sys

    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QApplication

    from potions.gui.common import set_rc_params
    from potions.gui.data import DEFAULT_CSV_PATH, load_input_data
    from potions.gui.window import MainWindow

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="potions-gui",
        description="East River lateral-model calibration GUI.",
    )
    parser.add_argument(
        "--csv",
        metavar="PATH",
        default=None,
        help=(
            "Path of the series CSV to load (default: the record "
            "shipped with the package)."
        ),
    )
    args: argparse.Namespace = parser.parse_args(argv)

    csv_path: Path = (
        Path(args.csv) if args.csv is not None else DEFAULT_CSV_PATH
    )

    set_rc_params()
    app: QApplication = QApplication([])
    app.setStyle("Fusion")
    # One point larger than the platform default, so the workbench
    # reads comfortably (no-op for pixel-sized platform fonts, whose
    # point size reads as -1).
    font: QFont = app.font()
    font.setPointSize(font.pointSize() + 1)
    app.setFont(font)

    input_data = load_input_data(csv_path)
    window: MainWindow = MainWindow(
        input_data,
        # The state records the session's CSV only when it is not
        # the record shipped with the package.
        csv_path=str(csv_path) if csv_path != DEFAULT_CSV_PATH else None,
    )
    window.show()
    sys.exit(app.exec())
# =================== #
