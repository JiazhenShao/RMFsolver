import hashlib
import importlib
import json
import signal
import sys
import time
import unittest
from pathlib import Path

ROOT = Path('/Users/janshao/NSM_related')
PREFIX = Path(sys.argv[1] if len(sys.argv) > 1 else '/private/tmp/nsm_analytic_baseline')
DIRECT_NAMES = '''
test_phase_velocity_public_api_surface
test_analytic_velocity_recovers_synthetic_equilibrium_and_uses_eos_density
test_analytic_velocity_lte_formula_and_default_mode
test_analytic_lte_a_0plus_stable_expression_and_slow_front_limit
test_analytic_lte_domain_is_checked_with_a_0plus_lte_not_A
test_analytic_A_matches_exact_eos_nK_over_nB
test_analytic_velocity_rejects_removed_interface_mode_selector
test_analytic_velocity_solves_hydro_jump_internally_without_jB_input
test_analytic_methods_accept_zero_upstream_temperature
test_analytic_methods_reject_negative_upstream_temperature
test_analytic_velocity_gamma_matches_weak_rate_and_seconds_shortcut
test_analytic_velocity_validates_xi_endpoint_and_density_domain
test_microphysics_helpers_use_precise_weak_rate
test_kaon_transport_rate_uses_baryon_prefactor_and_preserves_eos_diagnostics
test_microphysics_uses_shared_alpha_s_0p4_and_consistent_diffusion
test_all_rate_tagged_result_paths_identify_the_approximation
test_energy_microphysics_zero_temperature_uses_inverse_diffusion_limit
test_energy_microphysics_positive_temperature_computes_inverse_diffusion_directly
test_local_energy_closure_uses_absolute_nK_target
test_new_nK_energy_solver_public_signature
test_nK_tail_residual_retains_massive_equilibrium_offset
test_uNmax_default_jB_guess_matches_energy_nK_policy
test_solve_front_isothermal_warm_case_remains_finite
test_solve_front_energy_conserving_nK_regression
test_solve_front_energy_conserving_uNmax_regression
'''.split()
MODULES = [
    'test_phase_velocity_fixed_T_analytic',
    'test_phase_velocity_numerical_i2',
    'test_phase_velocity_isothermal_analytic',
    'test_phase_velocity_isothermal_a_0plus_max',
]
source_files = ['RMFsolver/phase_velocity.py', 'tests/test_phase_velocity.py'] + [f'tests/{name}.py' for name in MODULES]
metadata = {'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_files}, 'start_epoch': time.time(), 'timeout_seconds_per_test': 180, 'records': []}

def timeout_handler(signum, frame):
    raise TimeoutError('Baseline test exceeded 180 seconds')
signal.signal(signal.SIGALRM, timeout_handler)

class NamedFunctionTest(unittest.FunctionTestCase):
    def id(self):
        return 'test_phase_velocity.' + self._testFunc.__name__

class RecordingResult(unittest.TextTestResult):
    def startTest(self, test):
        self.test_started = time.monotonic()
        signal.alarm(180)
        super().startTest(test)
    def record(self, test, status, error=None):
        metadata['records'].append({'id': test.id(), 'status': status, 'seconds': time.monotonic() - self.test_started, 'error': error})
        PREFIX.with_suffix('.json').write_text(json.dumps(metadata, indent=2))
    def addSuccess(self, test):
        self.record(test, 'PASS')
        super().addSuccess(test)
    def addFailure(self, test, err):
        self.record(test, 'FAIL', self._exc_info_to_string(err, test))
        super().addFailure(test, err)
    def addError(self, test, err):
        self.record(test, 'ERROR', self._exc_info_to_string(err, test))
        super().addError(test, err)
    def stopTest(self, test):
        signal.alarm(0)
        super().stopTest(test)

module = importlib.import_module('test_phase_velocity')
import scipy
metadata['scipy_version'] = scipy.__version__
metadata['scipy_path'] = scipy.__file__
print(json.dumps({k: v for k, v in metadata.items() if k != 'records'}, indent=2), flush=True)
suite = unittest.TestSuite([NamedFunctionTest(getattr(module, name)) for name in DIRECT_NAMES])
for name in MODULES:
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(importlib.import_module(name)))
result = unittest.TextTestRunner(verbosity=2, resultclass=RecordingResult).run(suite)
metadata['tests_run'] = result.testsRun
metadata['passed'] = sum(record['status'] == 'PASS' for record in metadata['records'])
metadata['failed'] = len(result.failures)
metadata['errors'] = len(result.errors)
metadata['end_epoch'] = time.time()
PREFIX.with_suffix('.json').write_text(json.dumps(metadata, indent=2))
print(json.dumps({k: v for k, v in metadata.items() if k != 'records'}, indent=2), flush=True)
sys.exit(0 if result.wasSuccessful() else 1)
