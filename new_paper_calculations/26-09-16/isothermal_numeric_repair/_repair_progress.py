"""Display-only adapter: leave checkpoint-bound numerical workflow unchanged."""
from concurrent.futures import ThreadPoolExecutor, as_completed

from _repair_common import LEVELS
from _repair_search import Repair, ray_id


class ProgressRepair(Repair):
    def __init__(self, *args, bar_factory, **kwargs):
        self.bar_factory = bar_factory
        self.bar = None
        self.validation_seen = None
        self.evaluation_count = 0
        super().__init__(*args, progress=self, **kwargs)

    def update(self, count=1):
        """Called only after numerical evaluations have been checkpointed."""
        with self.lock:
            self.evaluation_count += count
            if self.bar is not None:
                self.bar.set_postfix(evals=self.evaluation_count, refresh=True)

    def run(self, worker_count, *, smoke=False):
        self.scheduled = 0
        self.initial_total = 4 if smoke else len(LEVELS) * len(self.theta)
        with self.bar_factory(total=self.initial_total, desc="Contour targets", unit="target") as bar:
            self.bar = bar
            try:
                return super().run(worker_count, smoke=smoke)
            finally:
                self.bar = None

    def _batch(self, tasks, worker_count):
        if not tasks:
            return
        self.scheduled += len(tasks)
        # Refinement/retry work is only known after earlier targets finish.
        self.bar.total = max(self.initial_total, self.scheduled)
        self.bar.refresh()
        with ThreadPoolExecutor(max_workers=min(worker_count, len(tasks))) as pool:
            futures = {pool.submit(self.solve_ray, level, theta): (level, theta)
                       for level, theta in tasks}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    level, theta = futures[future]
                    with self.lock:
                        self.payload["rays"][ray_id(level, theta)] = dict(
                            level=level, theta_rad=theta, roots=[], status="exception",
                            message=f"{type(exc).__name__}: {exc}")
                        self.save()
                # solve_ray persists its result before returning. Resumed successes
                # are already durable. An exception is persisted above as well.
                with self.lock:
                    self.bar.update(1)

    def save(self):
        super().save()
        if self.validation_seen is not None:
            for name, row in self.payload["validation"].items():
                if name not in self.validation_seen and row is not self.validation_before.get(name):
                    self.validation_seen.add(name)
                    self.bar.update(1)

    def validate_representatives(self):
        with self.bar_factory(total=4, desc="Convergence checks", unit="check") as bar:
            self.bar = bar
            self.validation_before = dict(self.payload["validation"])
            self.validation_seen = {name for name, row in self.validation_before.items()
                                    if row["status"] == "success"}
            bar.update(len(self.validation_seen))
            try:
                return super().validate_representatives()
            finally:
                self.validation_seen = None
                self.bar = None
