"""Killable calls adapted from 26-08-21/_radial_shell_continuation.py.

Same spawn/Pipe/terminate/kill protocol; cancellation also stops active calls.
"""
import importlib
import multiprocessing as mp
import time
import traceback


def _module_call_child(connection, module_name, function_name, args, kwargs):
    try:
        function = getattr(importlib.import_module(module_name), function_name)
        connection.send(("result", function(*args, **kwargs)))
    except BaseException as exc:
        connection.send(("error", type(exc).__name__, str(exc), traceback.format_exc(limit=8)))
    finally:
        connection.close()


def _stop_process(process, grace_s=2.0):
    if not process.is_alive():
        process.join(timeout=0.0)
        return
    process.terminate()
    process.join(timeout=grace_s)
    if process.is_alive():
        process.kill()
        process.join(timeout=grace_s)


def run_module_call_with_hard_timeout(module_name, function_name, *, args=(),
                                    kwargs=None, timeout_s, cancel=None):
    context = mp.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=_module_call_child,
                              args=(send, module_name, function_name, tuple(args), dict(kwargs or {})))
    process.start()
    send.close()
    deadline = time.monotonic() + timeout_s
    try:
        while not receive.poll(min(0.1, max(0, deadline - time.monotonic()))):
            if cancel is not None and cancel.is_set():
                raise InterruptedError("Calculation cancelled")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Analytical method exceeded {timeout_s:g} s")
        try:
            payload = receive.recv()
        except EOFError as exc:
            raise RuntimeError("Analytical child exited without a result") from exc
        process.join(timeout=2.0)
        if payload[0] == "result":
            return payload[1]
        raise RuntimeError(f"{payload[1]}: {payload[2]}\n{payload[3]}")
    finally:
        receive.close()
        if process.is_alive():
            _stop_process(process)
        else:
            process.join()
