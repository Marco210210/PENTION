# script_test/test_scenario.py
from GaussianPuff.ScenarioExample import make_config
from GaussianPuff.GaussianModel import run_dispersion_model

cfg, bounds = make_config()
C1, (x, y, z), times, *_ = run_dispersion_model(cfg, bounds=bounds)
print("OK:", C1.shape, x.shape, y.shape, len(times))
