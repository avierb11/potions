"""
Interactive GUI for `potions`.

The package provides one GUI: the East River lateral-model
calibration workbench (`potions.gui.window`).  It is a PySide6
desktop app: the model's figures in a single panel (flipped by
arrow buttons or the Left / Right keys: the hydrology, storage,
and partitioning figures, joined by the river concentrations,
river loads, concentration vs. discharge, zone concentrations,
and reaction rates figures in full mode, plus the gradient and
line search pages when those jobs have finished) over a row that holds the
fixed-size per-run objective-function table on the left and every
calibrated parameter on the right, grouped by zone (hydrology in
blue, reactive transport in red).  Parameter changes re-run the
models on a background thread; a *Save run* click stores the
run's model output and figures together with the session state,
which the app restores on its next start.

Start the GUI with the console script installed with the package::

    potions-gui

or programmatically with `main`.  The options are ``--csv`` (the
series CSV to load) and ``--network`` (a reaction-network YAML
file to run)::

    potions-gui --csv path/to/series.csv --network path/to/network.yaml
"""

from collections.abc import Sequence
from pathlib import Path

import potions as pt

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
            supported options are ``--csv``, the path of the
            series CSV to use (default: the record shipped
            with the package), and ``--network``, the path of a
            reaction-network YAML file to run (default: the East
            River SOC / DOC / calcite preset; a non-YAML path
            ends with an argparse usage error).
    """
    import argparse
    import potions as pt
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
    parser.add_argument(
        "--network",
        metavar="PATH",
        default=None,
        help=(
            "Path of a reaction-network YAML file to run (default: "
            "the East River SOC / DOC / calcite preset)."
        ),
    )
    args: argparse.Namespace = parser.parse_args(argv)

    csv_path: Path = (
        Path(args.csv) if args.csv is not None else DEFAULT_CSV_PATH
    )

    network: pt.ReactionNetwork | None = None
    if args.network is not None:
        network_path: Path = Path(args.network)
        if network_path.suffix.lower() not in (".yaml", ".yml"):
            parser.error(
                f"--network {args.network} must be a .yaml or .yml file."
            )
        network = _load_network_file(network_path)

    set_rc_params()
    app: QApplication = QApplication([])
    app.setStyle("Fusion")
    # A fixed 12-point base font: the workbench's smallest text is
    # the base font itself, so no text in the window is ever smaller
    # than 12 point (the platform default is usually smaller).
    font: QFont = app.font()
    font.setPointSize(12)
    app.setFont(font)

    input_data = load_input_data(csv_path, network=network)
    network_name: str | None = (
        Path(args.network).stem if args.network is not None else None
    )
    window: MainWindow = MainWindow(
        input_data,
        # The state records the session's CSV only when it is not
        # the record shipped with the package.
        csv_path=str(csv_path) if csv_path != DEFAULT_CSV_PATH else None,
        network_name=network_name,
    )
    window.show()
    sys.exit(app.exec())


def _load_network_file(path: Path) -> pt.ReactionNetwork:
    """
    Load a reaction-network YAML file for the ``--network`` option.

    Args:
        path: The path of the network file (checked for a YAML
            suffix by the caller).

    Returns:
        The parsed `ReactionNetwork` (V-gated by `from_file`).

    Raises:
        FileNotFoundError:
            Raised when the file does not exist.
        ValueError:
            Raised when the file is not valid network YAML or fails
            validation; the message names the file and the issue(s).
    """
    return pt.ReactionNetwork.from_file(str(path))
# =================== #
