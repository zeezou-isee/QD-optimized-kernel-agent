"""Backward-compat stub: forwards to cli/run_layer_oracle.py so old commands keep working.

Old: python opgen/run_layer_oracle.py ...
New: python opgen/cli/run_layer_oracle.py ... (also still works)
"""
import sys
from pathlib import Path
# Need EndtoEndMobilekernelAgent/ on sys.path so `import opgen` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "cli"))
import opgen as _o; _o.bootstrap_paths()
from run_layer_oracle import main  # cli/run_layer_oracle.py is on sys.path
if __name__ == "__main__":
    main()
