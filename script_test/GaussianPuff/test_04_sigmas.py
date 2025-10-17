# tests/test_sigmas.py
import numpy as np
from GaussianPuff.SigmaCalculation import calc_sigmas

def test_sigmas_basic_properties():
    x = np.linspace(1, 5000, 200)  # m
    for cat in [1,2,3,4,5,6]:
        sy, sz = calc_sigmas(cat, x)
        assert np.all(np.isfinite(sy)) and np.all(np.isfinite(sz))
        assert np.all(sy > 0) and np.all(sz > 0)
        # a grandi linee: sigma cresce con la distanza (non necessariamente strettamente)
        assert sy[-1] >= sy[0] and sz[-1] >= sz[0]
