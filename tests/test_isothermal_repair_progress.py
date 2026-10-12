"""Display-only checks: no EOS or BVP evaluations."""
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
    "new_paper_calculations/26-09-16/isothermal_numeric_repair"))
from _repair_progress import ProgressRepair
from _repair_search import Repair


class Bar:
    def __init__(self, **kwargs):
        self.total = kwargs["total"]
        self.n = 0
        self.postfix = {}
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def update(self, count=1): self.n += count
    def refresh(self): pass
    def set_postfix(self, **kwargs): self.postfix.update(kwargs)


class ProgressTests(unittest.TestCase):
    def repair(self):
        repair = ProgressRepair.__new__(ProgressRepair)
        repair.lock = threading.RLock()
        repair.theta = [0., 1., 2.]
        repair.payload = {"rays": {}, "validation": {}}
        repair.validation_seen = None
        repair.evaluation_count = 0
        repair.bar = None
        self.bars = []
        def factory(**kwargs):
            bar = Bar(**kwargs)
            self.bars.append(bar)
            return bar
        repair.bar_factory = factory
        return repair

    def test_full_target_total_grows_only_for_extra_work(self):
        repair = self.repair()
        saved = []
        def solve(level, theta):
            saved.append((level, theta))
            repair.update()
        repair.solve_ray = solve
        def run(instance, workers, smoke=False):
            instance._batch([(500., float(i)) for i in range(21)], workers)
            self.assertEqual((instance.bar.n, instance.bar.total), (21, 21))
            instance._batch([(1000., 1.)], workers)
        with patch.object(Repair, "run", run):
            repair.run(1)
        self.assertEqual((self.bars[0].n, self.bars[0].total), (22, 22))
        self.assertEqual(len(saved), 22)
        self.assertEqual(self.bars[0].postfix["evals"], 22)

    def test_exception_is_saved_before_target_increment(self):
        repair = self.repair()
        repair.bar = Bar(total=21)
        repair.initial_total = 21
        repair.scheduled = 0
        def fail(*args): raise RuntimeError("synthetic failure")
        repair.solve_ray = fail
        observed = []
        with patch.object(Repair, "save", lambda obj: observed.append(obj.bar.n)):
            repair._batch([(500., 0.)], 1)
        self.assertEqual(observed, [0])
        self.assertEqual(repair.bar.n, 1)

    def test_validation_counts_reused_and_new_results(self):
        repair = self.repair()
        repair.payload["validation"] = {"a": {"status": "success"}, "b": {"status": "unresolved"}}
        def validate(instance):
            for name in ("b", "c", "d"):
                instance.payload["validation"][name] = {"status": "unresolved"}
                instance.save()
        with patch.object(Repair, "save"), patch.object(Repair, "validate_representatives", validate):
            repair.validate_representatives()
        self.assertEqual((self.bars[0].n, self.bars[0].total), (4, 4))


if __name__ == "__main__":
    unittest.main()
