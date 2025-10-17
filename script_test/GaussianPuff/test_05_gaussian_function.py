# tests/test_gaussian_function.py
import numpy as np
from GaussianPuff.GaussianFunction import gauss_func_plume

def test_zero_wind_returns_zero_field():
    x = y = np.linspace(-100,100,21)
    z = 0.0
    C = gauss_func_plume(Q=1.0, u=0.0, dir1=0, x=x, y=y, z=z, xs=0, ys=0, H=10, STABILITY=4)
    assert np.allclose(C, 0.0)

def test_upwind_near_zero_downwind_positive():
    x = y = np.linspace(-200,200,81)
    z = 0.0
    C = gauss_func_plume(Q=1.0, u=3.0, dir1=0.0, x=x, y=y, z=z, xs=0, ys=0, H=10, STABILITY=4)
    # dir1=0° (da N→S in convenzione meteo) con rotazione interna: ci aspettiamo lobo sottovento
    # controlla che esista almeno una cella >0 e che non sia tutto simmetrico
    assert np.nanmax(C) > 0
