import sys
from pathlib import Path

# The probes are scripts importing each other as top-level modules (`_common`,
# `calibration`, `p14_laya_temperature`), and the bench reuses them as they are.
sys.path.append(str(Path(__file__).resolve().parent.parent / "probes"))
