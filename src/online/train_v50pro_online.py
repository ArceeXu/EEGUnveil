"""Legacy filename redirected to the OpenBMI holdout training CLI.

The GUI loads existing offline artifacts; no separate online training is needed.
Arguments are identical to src/offline/run_openbmi_holdout.py.
"""
import sys
from pathlib import Path
from torch_bootstrap import prepare_torch_dlls

prepare_torch_dlls()
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.offline.run_openbmi_holdout import main

if __name__ == "__main__":
    main()

