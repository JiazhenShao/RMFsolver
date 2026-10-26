#!/usr/bin/env python3
"""Cluster exact-cold analytical contours; workers follow the CPU allocation."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'shared'))
from cold_scan import cluster_workers, main

if __name__ == '__main__':
    raise SystemExit(main(default_workers=cluster_workers(),
                         default_output=HERE/'adiabatic-T0plus0-xi-minus0p85.npy'))
