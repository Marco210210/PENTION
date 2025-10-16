from fastapi import FastAPI
import os, sys
from pydantic import BaseModel, Field
from typing import List, Tuple, Optional
import logging
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from GaussianPuff.GaussianModel import run_dispersion_model
from GaussianPuff.Config import (
    ModelConfig, WindType, StabilityType, PasquillGiffordStability,
    NPS, OutputType, DispersionModelType   # <— aggiungi DispersionModelType
)

# Configurazione logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

class ModelConfigRequest(BaseModel):
    days: int
    RH: float
    aerosol_type: str
    humidify: bool
    stability_profile: str
    stability_value: str
    wind_type: str
    wind_speed: float
    output: str
    stacks: List[Tuple[float, float, float, float]]
    dry_size: Optional[float] = 60e-9
    x_slice: Optional[int] = 26
    y_slice: Optional[int] = 1
    grid_size: Optional[int] = 100
    dispersion_model: str
    config_puff: Optional[dict] = None

# Model per il payload (fix Field)
class Payload(BaseModel):
    config: ModelConfigRequest
    bounds: List[float] = Field(min_items=4, max_items=4)
    seed: Optional[int] = None
    return_field: bool = False  # se True, restituisce C1 intero (occhio a dimensioni!)


app = FastAPI()

@app.post("/start_simulation")
def start_simulation(payload: Payload):
    logger.info("Ricevuta richiesta /start_simulation")
    try:
        # --- NEW: imposta il seed se presente ---
        import random
        seed = payload.seed
        seed_val = None
        if seed is not None:
            try:
                seed_val = int(seed)
                np.random.seed(seed_val)
                random.seed(seed_val)
                logger.info(f"Seed impostato a {seed_val}")
            except Exception as e:
                logger.warning(f"Impossibile impostare il seed: {e}")

        raw_config = payload.config.model_dump()
        logger.info(raw_config)

        wind_type= WindType.from_string(raw_config["wind_type"])
        stability_type = StabilityType.from_string(raw_config["stability_profile"])
        output_type= OutputType.from_string(raw_config["output"])
        stability_value=PasquillGiffordStability.from_string(raw_config["stability_value"])
        nps_type= NPS.from_string(raw_config["aerosol_type"])
        dispersion_model = DispersionModelType.from_string(raw_config["dispersion_model"])

        config = ModelConfig(
            days=raw_config["days"],
            RH=raw_config["RH"],
            aerosol_type=nps_type,
            humidify=raw_config["humidify"],
            stability_profile=stability_type,
            stability_value=stability_value,
            wind_type=wind_type,
            wind_speed=raw_config["wind_speed"],
            output=output_type,
            stacks=raw_config["stacks"],
            dry_size=raw_config.get("dry_size", 60e-9),
            x_slice=raw_config.get("x_slice", 26),
            y_slice=raw_config.get("y_slice", 1),
            grid_size=raw_config.get("grid_size", 100),
            dispersion_model=dispersion_model,                 # <— corretto
            config_puff=raw_config.get("config_puff", None),
            seed=seed_val if seed is not None else 42          # <— passiamo davvero il seed
        )

        bounds = payload.bounds
        logger.info(f"Configurazione modello creata: {config}")
        logger.info(f"Bounds ricevuti: {bounds}")

        result = run_dispersion_model(config, bounds)
        logger.info("Simulazione completata")

        C1, (x, y, z), times, stability, wind_dir, stab_label, wind_label, puff = result
        logger.info("End gaussian model simulation")

        
        # risposta “slim” di default
        return_field = payload.return_field
        meta = {
            "status": 200,
            "shape": (len(x), len(y), len(times)),
            "x": [float(x[0]), float(x[-1]), len(x)],
            "y": [float(y[0]), float(y[-1]), len(y)],
            "z_levels": len(z),
            "times_hours": len(times),
            "stability_label": str(stab_label),
            "wind_label": str(wind_label),
            "seed": config.seed,
        }

        if return_field:
            meta["concentration"] = C1.tolist()  # attenzione: può essere enorme
        else:
            meta["concentration_min"] = float(np.nanmin(C1))
            meta["concentration_max"] = float(np.nanmax(C1))

        return meta

    except Exception as error:
        logger.exception("Errore durante la simulazione")
        raise error