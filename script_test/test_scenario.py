# script_test/test_scenario.py
from gaussianPuff.scenario_example import make_config
from gaussianPuff.gaussianModel import run_dispersion_model

cfg, bounds = make_config()
C1, (x, y, z), times, *_ = run_dispersion_model(cfg, bounds=bounds)
print("OK:", C1.shape, x.shape, y.shape, len(times))
