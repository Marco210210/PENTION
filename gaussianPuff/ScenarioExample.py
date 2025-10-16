# gaussianPuff/scenario_example.py
# =============================================================================
# SCENARIO DI SIMULAZIONE PENTION-S
# Qui modifichi solo i parametri sotto; tutto il resto del progetto li userà automaticamente.
# =============================================================================

from GaussianPuff.Config import (
    ModelConfig, WindType, StabilityType, PasquillGiffordStability,
    NPS, OutputType, DispersionModelType, ConfigPuff
)

# === PARAMETRI GENERALI ======================================================
DAYS            = 1                 # Durata simulazione (in giorni)
GRID_SIZE       = 100               # Risoluzione griglia (nx=ny=GRID_SIZE)
BOUNDS          = (-250, -250, 250, 250)  # Estensione dell’area [xmin, ymin, xmax, ymax] in metri
# =============================================================================

# === PARAMETRI METEO =========================================================
WIND_SPEED      = 3.0               # Velocità media del vento [m/s]
WIND_TYPE       = WindType.CONSTANT # Tipologia vento: CONSTANT | FLUCTUATING | PREVAILING

STABILITY_TYPE  = StabilityType.CONSTANT  # Tipo di profilo di stabilità: CONSTANT | ANNUAL
STABILITY_VALUE = PasquillGiffordStability.NEUTRAL  # Classe PG: VERY_UNSTABLE(1) ... VERY_STABLE(6)

RH              = 0.50              # Umidità relativa [0..1]
HUMIDIFY        = False             # True: attiva crescita igroscopica (solo se RH>0)
# =============================================================================

# === PARAMETRI SORGENTI ======================================================
# Lista di sorgenti (x_s, y_s, Q, H)
#  x_s, y_s = posizione [m]
#  Q = tasso di emissione relativo (più grande = più concentrazione)
#  H = altezza sorgente [m]
STACKS = [
    (0.0, 0.0, 1.0, 10.0),     # Sorgente principale (centro, 10 m)
    # (100.0, -50.0, 0.5, 5.0) # Esempio seconda sorgente
]

# Tipo di sostanza simulata (scegli uno dei seguenti):
# NPS.CANNABINOID_ANALOGUES
# NPS.CATHINONE_ANALOGUES
# NPS.PHENETHYLAMINE_ANALOGUES
# NPS.PIPERAZINE_ANALOGUES
# NPS.TRYPTAMINE_ANALOGUES
# NPS.FENTANYL_ANALOGUES
# NPS.OTHER_COMPOUNDS
AEROSOL_TYPE = NPS.CATHINONE_ANALOGUES
# =============================================================================

# === PARAMETRI MODELLO DI DISPERSIONE =======================================
DISPERSION = DispersionModelType.PLUME   # PLUME | PUFF
PUFF_CFG   = ConfigPuff(puff_interval=1, max_puff_age=6)  # valido solo se DISPERSION=PUFF
# =============================================================================

# === OUTPUT & RANDOMNESS =====================================================
OUTPUT = OutputType.NO_PLOT   # PLAN_VIEW | HEIGHT_SLICE | SURFACE_TIME | NO_PLOT
SEED   = 42                   # Seed per random (riproducibilità)
# =============================================================================


def make_config() -> tuple[ModelConfig, tuple[float, float, float, float]]:
    """Ritorna (config, bounds) pronto per run_dispersion_model()."""
    cfg = ModelConfig(
        days=DAYS,
        RH=RH,
        aerosol_type=AEROSOL_TYPE,
        humidify=HUMIDIFY,
        stability_profile=STABILITY_TYPE,
        stability_value=STABILITY_VALUE,
        wind_type=WIND_TYPE,
        wind_speed=WIND_SPEED,
        output=OUTPUT,
        stacks=STACKS,
        grid_size=GRID_SIZE,
        dispersion_model=DISPERSION,
        config_puff=(PUFF_CFG if DISPERSION == DispersionModelType.PUFF else None),
        seed=SEED,
    )
    return cfg, BOUNDS
