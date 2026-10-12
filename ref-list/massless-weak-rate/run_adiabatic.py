#!/usr/bin/env python3
"""Reproduce the upstream scan, conserved trajectories, checks and figures."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--force-quadrature',action='store_true',
                        help='Recompute collision integrals after changing their source')
    args=parser.parse_args()
    here=Path(__file__).resolve().parent
    for script in ('adiabatic_upstream.py','adiabatic_sensitivity.py','complete_adiabatic_quadrature.py',
                   'adiabatic_verify.py','adiabatic_report.py'):
        command=[sys.executable,str(here/script)]
        if script=='adiabatic_sensitivity.py' and args.force_quadrature:
            command.append('--force')
        subprocess.run(command,cwd=here,check=True)


if __name__=='__main__':
    main()
