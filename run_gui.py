"""
Entry script for the East River lateral-model calibration GUI.

A thin wrapper around `potions.gui.main` so the GUI can be launched
from the repository root with ``python run_gui.py`` (optionally
``--csv PATH``); the installed `potions-gui` console script provides
the same entry point.
"""

from potions.gui import main

if __name__ == "__main__":
    main()
