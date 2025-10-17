# tests/test_sensor_sampling.py
import numpy as np
from GaussianPuff.ScenarioExample import make_config
from GaussianPuff.GaussianModel import run_dispersion_model
from GaussianPuff.Sensor import SensorSubstance

def test_sensor_sampling_timeseries():
    cfg, bounds = make_config()
    C,(x,y,z),times,*_ = run_dispersion_model(cfg, bounds)
    s = SensorSubstance(sensor_id=1, x=0.0, y=0.0, z=2.0, noise_level=0.0)
    s.sample_substance(C, x, y, times)
    assert s.times is not None and len(s.times) == len(times)
    assert s.concentrations.shape == (len(times),)
    # niente NaN, valori finiti
    assert np.isfinite(s.concentrations).all()
