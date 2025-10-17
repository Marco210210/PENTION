# tests/test_seed_determinism.py
import numpy as np
from GaussianPuff.ScenarioExample import make_config
from GaussianPuff.GaussianModel import run_dispersion_model
from GaussianPuff.Config import WindType

def test_seed_determinism_fluctuating():
    cfg, bounds = make_config()
    cfg.wind_type = WindType.FLUCTUATING
    cfg.seed = 123
    C1, *_ = run_dispersion_model(cfg, bounds)
    C2, *_ = run_dispersion_model(cfg, bounds)
    assert np.allclose(C1, C2)
