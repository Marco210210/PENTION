from gaussianPuff.config import ModelConfig, WindType, StabilityType, PasquillGiffordStability, NPS, OutputType, DispersionModelType, ConfigPuff
from gaussianPuff.gaussianModel import run_dispersion_model

cfg = ModelConfig(
    days=1,
    RH=0.5,
    aerosol_type=NPS.CATHINONE_ANALOGUES,
    humidify=False,
    stability_profile=StabilityType.CONSTANT,
    stability_value=PasquillGiffordStability.NEUTRAL,
    wind_type=WindType.CONSTANT,
    wind_speed=3.0,
    output=OutputType.NO_PLOT,
    stacks=[(0.0, 0.0, 1.0, 10.0)],
    grid_size=50,
    dispersion_model=DispersionModelType.PLUME,
)

C1, (x, y, z), times, stability, wind_dir, *_ = run_dispersion_model(cfg, bounds=(-250, -250, 250, 250))
print(C1.shape, x.shape, y.shape, len(times))
