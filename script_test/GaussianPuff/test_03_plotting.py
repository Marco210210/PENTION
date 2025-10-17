# tests/test_plotting.py
import matplotlib
matplotlib.use("Agg")
from GaussianPuff.ScenarioExample import make_config
from GaussianPuff.GaussianModel import run_dispersion_model
from GaussianPuff.plot_utils import plot_plan_view, plot_surface_view_3d

def test_plot_plan_view_no_crash(tmp_path):
    cfg, bounds = make_config()
    cfg.output = cfg.output.NO_PLOT  # il plot legge solo i dati
    C,(x,y,z),times,stability,wind_dir,stab_label,wind_label,puffs = run_dispersion_model(cfg, bounds)
    # deve accettare assi 1D e trasporre internamente
    plot_plan_view(C, x, y, "test", wind_dir=wind_dir, wind_speed=cfg.wind_speed)
    # se arrivi qui, non è crashato

def test_plot_3d_no_crash(tmp_path):
    cfg, bounds = make_config()
    C,(x,y,z),times,stability,wind_dir,stab_label,wind_label,puffs = run_dispersion_model(cfg, bounds)
    plot_surface_view_3d(C, x, y, z=z, times=times, t_index=0, title="ok")
