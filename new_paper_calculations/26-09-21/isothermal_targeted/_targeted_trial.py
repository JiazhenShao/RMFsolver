"""Disposable numerical trials with independent parent and child deadlines."""
import math
import multiprocessing as mp
import signal
import time


def _numerical_trial(**kwargs):
    # Import the unchanged validator only after the child alarm is armed.
    from _repair_common import numerical_trial
    return numerical_trial(**kwargs)


def _child(connection, deadline, trial, kwargs):
    # The kernel's default signal action also interrupts native numerical code,
    # and remains effective if the controlling parent is abruptly terminated.
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.setitimer(signal.ITIMER_REAL, max(.001, deadline - time.monotonic()))
    try:
        try:
            result = trial(**kwargs)
        except Exception as exc:
            result = dict(success=False, status="exception",
                          message=f"{type(exc).__name__}: {exc}")
        connection.send(result)
    finally:
        connection.close()


def hard_trial(*, timeout, **kwargs):
    """Run the unchanged numerical trial; startup counts toward the deadline."""
    timeout = float(timeout)
    if not math.isfinite(timeout) or timeout <= 0.:
        raise ValueError("Trial timeout must be finite and positive")
    start = time.monotonic()
    deadline = start + timeout
    context = mp.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=_child, args=(send, deadline, _numerical_trial, kwargs))
    result = None
    started = False
    exitcode = None
    try:
        process.start()
        started = True
        send.close()
        if receive.poll(max(0., deadline - time.monotonic())):
            try:
                result = receive.recv()
            except (EOFError, OSError):
                result = dict(success=False, status="child_error",
                              message="Worker exited without a complete result")
        else:
            result = dict(success=False, status="trial_timeout",
                          message=f"Hard trial limit: {timeout:g} s")
    finally:
        receive.close()
        send.close()
        if started:
            process.join(timeout=.05)
            if process.is_alive():
                process.terminate()
                process.join(timeout=.1)
            if process.is_alive():
                process.kill()
                process.join(timeout=.1)
            exitcode = process.exitcode
            if not process.is_alive():
                process.close()
    if result["status"] == "child_error":
        if exitcode == -signal.SIGALRM:
            result.update(status="trial_timeout", message=f"Child trial limit: {timeout:g} s")
        else:
            result["message"] += f" (exit code {exitcode})"
    result["runtime_s"] = time.monotonic() - start
    return result
