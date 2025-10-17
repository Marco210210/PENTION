# tests/test_smoke_gaussian.py
import numpy as np
from GaussianPuff.ScenarioExample import make_config
from GaussianPuff.GaussianModel import run_dispersion_model

def test_smoke_scenario_example():
    cfg, bounds = make_config()
    C, (x, y, z), times, stability, wind_dir, stab_label, wind_label, puff = run_dispersion_model(cfg, bounds)
    # Shape attesa da ScenarioExample: GRID_SIZE=100, DAYS=1 → 24 step
    assert C.shape == (len(x), len(y), len(times)) == (100, 100, 24)
    # Assi 1D ordinati strettamente crescenti
    assert np.all(np.diff(x) > 0) and np.all(np.diff(y) > 0)
    # z è livello suolo → array o scalare coerente
    assert z is not None
    # campione finito e non tutto zero
    assert np.isfinite(C).all()
    assert np.nanmax(C) > 0
