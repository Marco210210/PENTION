# script_test/GaussianPuff/test_api.py
import pytest, numpy as np
from httpx import AsyncClient, ASGITransport
from GaussianPuff.api_gaussian import app

@pytest.mark.asyncio
async def test_start_simulation_endpoint():
    payload = {
        "config": {
            "days": 1, "RH": 0.5, "aerosol_type": "CATHINONE_ANALOGUES",
            "humidify": False, "stability_profile": "CONSTANT",
            "stability_value": "NEUTRAL",
            "wind_type": "CONSTANT", "wind_speed": 3.0,
            "output": "NO_PLOT",
            "stacks": [[0.0, 0.0, 1.0, 10.0]],
            "grid_size": 100,
            "dispersion_model": "PLUME"
        },
        "bounds": [-250, -250, 250, 250],
        "seed": 42,
        "return_field": False
    }

    # >>> cambia qui:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.post("/start_simulation", json=payload)

    assert r.status_code == 200
    j = r.json()
    assert j["shape"] == [100, 100, 24]
    assert j["times_hours"] == 24
    assert np.isfinite(j["concentration_min"]) and np.isfinite(j["concentration_max"])
