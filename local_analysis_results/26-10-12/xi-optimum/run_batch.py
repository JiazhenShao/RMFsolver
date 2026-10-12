"""Evaluate both analytical methods at every benchmark point (resumable JSONL)."""
import json, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import common

XI = float(sys.argv[1]) if len(sys.argv) > 1 else -0.85
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
OUT = f"analytical_xi{XI:+.4f}.jsonl"
KEYS = sys.argv[3].split(",") if len(sys.argv) > 3 else None


def clean(obj):
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


if __name__ == "__main__":
    points = common.benchmark_points()
    done = set()
    try:
        with open(OUT) as handle:
            done = {json.loads(line)["key"] for line in handle}
    except FileNotFoundError:
        pass
    todo = [p for k, p in points.items() if k not in done and (KEYS is None or k in KEYS)]
    print(f"xi={XI}: {len(todo)} to do, {len(done)} done", flush=True)
    started = time.time()
    with ProcessPoolExecutor(WORKERS) as pool, open(OUT, "a") as handle:
        futures = {pool.submit(common.evaluate, p, XI): p["key"] for p in todo}
        for n, future in enumerate(as_completed(futures), 1):
            row = future.result()
            handle.write(json.dumps(clean(row)) + "\n")
            handle.flush()
            if n % 25 == 0 or n == len(todo):
                print(f"{n}/{len(todo)} {time.time() - started:.0f}s", flush=True)
