"""Run frozen existing test selections without editing solver or tests.

Usage: python -B run_regression.py {baseline,after} GROUP
GROUP is historical_60, analytic_slow, or slow_bvp. Historical identifiers
retain existing legacy public/test names for compatibility only.
"""
import argparse
import hashlib
import importlib
import json
import signal
import sys
import time
import unittest
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("phase")
parser.add_argument("group", choices=("historical_60", "analytic_slow", "slow_bvp"))
parser.add_argument("--timeout", type=int, default=180)
parser.add_argument("--selection-file", type=Path, default=DIRECTORY / "regression_selection.json")
parser.add_argument("--source-root", type=Path, default=ROOT)
args = parser.parse_args()
args.source_root = args.source_root.resolve()
sys.path.insert(0, str(args.source_root))
selection = json.loads(args.selection_file.read_text())[args.group]
prefix = DIRECTORY / f"{args.phase}_{args.group}"
log = prefix.with_suffix(".log").open("w", buffering=1)
sys.stdout = log
sys.stderr = log
metadata = {"phase": args.phase, "group": args.group, "selection": selection,
            "timeout_seconds_per_test": args.timeout, "records": [],
            "started_epoch": time.time(), "source_sha256": {},
            "source_root": str(args.source_root), "selection_file": str(args.selection_file)}
for file in ["RMFsolver/phase_velocity.py"] + sorted({"tests/" + item.split(".")[0] + ".py" for item in selection}):
    source_root = args.source_root if file.startswith("RMFsolver/") else ROOT
    metadata["source_sha256"][file] = hashlib.sha256((source_root / file).read_bytes()).hexdigest()

def save():
    prefix.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")

class TestTimeout(BaseException):
    """Avoid an implementation's broad Exception handler swallowing timeout."""

def timeout_handler(signum, frame):
    raise TestTimeout(f"Test exceeded {args.timeout} seconds")

signal.signal(signal.SIGALRM, timeout_handler)

class NamedFunctionTest(unittest.FunctionTestCase):
    def id(self):
        return self._testFunc.__module__ + "." + self._testFunc.__name__

suite = unittest.TestSuite()
for identifier in selection:
    parts = identifier.split(".")
    module = importlib.import_module(parts[0])
    if len(parts) == 2:
        suite.addTest(NamedFunctionTest(getattr(module, parts[1])))
    else:
        suite.addTest(getattr(module, parts[1])(parts[2]))
import scipy
import RMFsolver.phase_velocity as pv
metadata["scipy_version"] = scipy.__version__
metadata["scipy_path"] = scipy.__file__
metadata["phase_velocity_import_path"] = pv.__file__
metadata["imports_complete_epoch"] = time.time()
save()
print(json.dumps({key: value for key, value in metadata.items() if key != "records"}, indent=2), flush=True)

class RecordingResult(unittest.TextTestResult):
    def startTest(self, test):
        self.started = time.monotonic()
        self.current_errors = []
        self.current_status = "PASS"
        signal.setitimer(signal.ITIMER_REAL, args.timeout, args.timeout)
        super().startTest(test)
    def addFailure(self, test, err):
        self.current_status = "FAIL"
        self.current_errors.append(self._exc_info_to_string(err, test))
        super().addFailure(test, err)
    def addError(self, test, err):
        self.current_status = "ERROR"
        self.current_errors.append(self._exc_info_to_string(err, test))
        super().addError(test, err)
    def addSubTest(self, test, subtest, err):
        if err is not None:
            self.current_status = "FAIL" if issubclass(err[0], test.failureException) else "ERROR"
            self.current_errors.append(str(subtest) + "\n" + self._exc_info_to_string(err, test))
        super().addSubTest(test, subtest, err)
    def stopTest(self, test):
        signal.setitimer(signal.ITIMER_REAL, 0)
        metadata["records"].append({"id": test.id(), "status": self.current_status,
                                    "seconds": time.monotonic() - self.started,
                                    "error": "\n".join(self.current_errors) or None})
        save()
        super().stopTest(test)

result = unittest.TextTestRunner(verbosity=2, resultclass=RecordingResult).run(suite)
metadata.update(tests_run=result.testsRun, passed=sum(r["status"] == "PASS" for r in metadata["records"]),
                failed=sum(r["status"] == "FAIL" for r in metadata["records"]),
                errors=sum(r["status"] == "ERROR" for r in metadata["records"]),
                finished_epoch=time.time())
save()
print(json.dumps({key: value for key, value in metadata.items() if key not in ("records", "selection")}, indent=2), flush=True)
log.close()
sys.exit(0 if result.wasSuccessful() else 1)
