# tests/test_map_plot_optional.py
from GaussianPuff.ScenarioExample import make_config
from GaussianPuff.GaussianModel import run_dispersion_model
from GaussianPuff.plot_utils import plot_puff_on_map

def test_plot_puff_on_map_no_crash():
    cfg, bounds = make_config()
    C,(x,y,z),times,*_ = run_dispersion_model(cfg, bounds)
    # centro fittizio (lat, lon)
    m = plot_puff_on_map(C, x, y, center_lat=45.0, center_lon=9.0, timestep=-1, cutoff_norm=0.2)
    assert m is not None
