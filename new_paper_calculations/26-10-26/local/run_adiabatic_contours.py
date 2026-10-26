#!/usr/bin/env python3
"""Local exact-cold analytical contours; four workers by default."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'shared'))
from cold_scan import main

if __name__ == '__main__':
    raise SystemExit(main(default_workers=4,
                         default_output=HERE/'adiabatic-T0plus0-xi-minus0p85.npy'))
