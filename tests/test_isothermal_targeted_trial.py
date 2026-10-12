"""Synthetic spawned workers only: no EOS imports or numerical solves."""
import importlib
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import time
import unittest
from unittest.mock import patch


WORK = Path(__file__).resolve().parents[1] / "new_paper_calculations/26-09-21/isothermal_targeted"
sys.path.insert(0, str(WORK))
try:
    guarded = importlib.import_module("_targeted_trial")
except ModuleNotFoundError:
    guarded = None


def sleeping_trial(**kwargs):
    time.sleep(1.)
    return dict(success=True, status="success")


def successful_trial(**kwargs):
    return dict(success=True, status="success", echoed=kwargs["value"])


def failed_trial(**kwargs):
    raise ValueError("synthetic failure")


def empty_trial(**kwargs):
    os._exit(7)


class TargetedTrialTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(guarded, "The targeted child lifetime guard is not implemented")

    def test_parent_watchdog_returns_timeout_and_reaps_worker(self):
        before = {child.pid for child in mp.active_children()}
        start = time.monotonic()
        with patch.object(guarded, "_numerical_trial", sleeping_trial):
            result = guarded.hard_trial(timeout=.05)
        self.assertEqual(result["status"], "trial_timeout")
        self.assertFalse(result["success"])
        self.assertLess(time.monotonic() - start, .8)
        self.assertEqual({child.pid for child in mp.active_children()}, before)

    def test_child_alarm_terminates_without_parent_watchdog(self):
        context = mp.get_context("spawn")
        receive, send = context.Pipe(duplex=False)
        child = context.Process(target=guarded._child,
                                args=(send, time.monotonic() + .10, sleeping_trial, {}))
        try:
            child.start()
            send.close()
            child.join(timeout=.8)
            self.assertFalse(child.is_alive(), "Child must enforce its own deadline")
            self.assertEqual(child.exitcode, -signal.SIGALRM)
            self.assertTrue(receive.poll(.05))
            with self.assertRaises(EOFError):
                receive.recv()
        finally:
            receive.close()
            send.close()
            if child.is_alive():
                child.kill()
                child.join(timeout=.2)
            child.close()

    def test_success_and_keyword_arguments_survive_spawn(self):
        with patch.object(guarded, "_numerical_trial", successful_trial):
            result = guarded.hard_trial(timeout=.8, value=42)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["echoed"], 42)
        self.assertGreater(result["runtime_s"], 0.)

    def test_exception_is_returned_as_diagnostic(self):
        with patch.object(guarded, "_numerical_trial", failed_trial):
            result = guarded.hard_trial(timeout=.8)
        self.assertEqual(result["status"], "exception")
        self.assertIn("ValueError: synthetic failure", result["message"])

    def test_empty_pipe_reports_child_exit(self):
        with patch.object(guarded, "_numerical_trial", empty_trial):
            result = guarded.hard_trial(timeout=.8)
        self.assertEqual(result["status"], "child_error")
        self.assertIn("7", result["message"])


if __name__ == "__main__":
    unittest.main()
