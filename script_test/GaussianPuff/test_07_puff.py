# script_test/GaussianPuff/test_07_puff.py
import numpy as np
from GaussianPuff.ScenarioExample import make_config
from GaussianPuff.Config import DispersionModelType, ConfigPuff
from GaussianPuff.GaussianModel import run_dispersion_model

def test_puff_branch_runs():
    cfg, bounds = make_config()
    cfg.dispersion_model = DispersionModelType.PUFF
    cfg.config_puff = ConfigPuff(puff_interval=1, max_puff_age=6)

    # >>> aggiungi queste due righe:
    cfg.wind_speed = 0.005  # ~18 m/ora: i puff restano dentro la griglia
    # in alternativa: bounds = (-2500, -2500, 2500, 2500)

    C,(x,y,z),times,stability,wind_dir,stab_label,wind_label,puffs = run_dispersion_model(cfg, bounds)
    assert C.shape == (len(x), len(y), len(times))
    assert np.nanmax(C) > 0
