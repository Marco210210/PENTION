import os
import sys
import numpy as np
import random
from collections import Counter
import requests
import streamlit as st
from streamlit_folium import st_folium
from plot_functions import *
from utils import *
st.set_page_config(page_title="PentionSystem", layout="wide")

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from gaussianPuff.Sensor import SensorSubstance, SensorAir
from gaussianPuff.config import NPS, OutputType, DispersionModelType, ModelConfig

# --- in cima, subito dopo gli import ---
# Stato e helper
if 'progress' not in st.session_state:
    st.session_state['progress'] = 0
if 'run_in_progress' not in st.session_state:
    st.session_state['run_in_progress'] = False

def set_progress(v: int):
    st.session_state['progress'] = int(v)
    progress_bar.progress(st.session_state['progress'])

@st.cache_data(show_spinner=False)
def cached_binary_map(payload_dict: dict):
    r = requests.post(f"{CORRECTION_URL}/generate_binary_map", json=payload_dict)
    r.raise_for_status()
    return r.json()                     # ritorna direttamente il json

CORRECTION_URL   = os.environ.get("CORRECTION_URL",   "http://correction_dispersion:8001")
GAUSSIAN_URL     = os.environ.get("GAUSSIAN_URL",     "http://gaussian_dispersion_model:8002")
LOCALIZATION_URL = os.environ.get("LOCALIZATION_URL", "http://loc_emission_source:8003")
CLASSIFIER_URL   = os.environ.get("CLASSIFIER_URL",   "http://clas_nps:8000")

def coerce_grid_axes(C, x_in, y_in, t_in):
    """
    Converte x,y in assi 1D strettamente crescenti compatibili con C.shape,
    gestendo il caso in cui x,y arrivino come meshgrid 'flattened' (nx*ny).
    Garantisce t strettamente crescente e riallinea C lungo l'asse temporale se serve.

    Attesi:
      C.shape == (nx, ny, nt)
      x_in.size in {nx, nx*ny}
      y_in.size in {ny, nx*ny}
      t_in.size == nt (o riordinabile)

    Ritorna: C2, x_axis, y_axis, t_axis
    """
    C = np.asarray(C)
    x_in = np.asarray(x_in).ravel()
    y_in = np.asarray(y_in).ravel()
    t_in = np.asarray(t_in).ravel()

    if C.ndim != 3:
        raise ValueError(f"Concentration map must be 3D (x,y,t), got shape {C.shape}")

    nx, ny, nt = C.shape

    # ---- costruisci assi x,y da input che può essere [nx] o [nx*ny]
    # arrotondo per evitare duplicati numerici quasi-identici
    def axis_from_flat(arr, target_len, name):
        if arr.size == target_len:
            axis = arr
        elif arr.size == nx * ny:
            axis = np.unique(np.round(arr, 6))
        else:
            raise ValueError(f"{name} unexpected size {arr.size}; "
                             f"expected {target_len} or {nx*ny}")
        # forza stretta crescita (unique è già crescente)
        if axis.size != target_len:
            raise ValueError(f"{name} unique len {axis.size} != expected {target_len}")
        return axis

    x_axis = axis_from_flat(x_in, nx, "x")
    y_axis = axis_from_flat(y_in, ny, "y")

    # ---- tempo: assicura ordine stretto e riallinea C sull'asse t se serve
    # se ci sono duplicati o ordine non crescente, riordino
    if t_in.size != nt:
        raise ValueError(f"t len {t_in.size} != expected {nt}")

    # ordino t e applico lo stesso ordine a C lungo l'asse temporale
    order_t = np.argsort(t_in)
    if not np.all(order_t == np.arange(nt)):
        C = C[:, :, order_t]
        t_axis = t_in[order_t]
    else:
        t_axis = t_in

    # quick check su monotonicità stretta (se t ha duplicati, li rendo leggermente unici)
    dt = np.diff(t_axis)
    if not np.all(dt > 0):
        # rimuovo duplicati mantenendo ordine
        t_axis_unique, idx = np.unique(t_axis, return_index=True)
        t_axis = t_axis_unique
        C = C[:, :, np.sort(idx)]
        if t_axis.size != C.shape[2]:
            raise ValueError("Temporal axis dedup led to mismatch with C.")

    return C, x_axis, y_axis, t_axis

def run_application(payload):
    try:
        n_sensors = payload.get("Number of sensors", 10)
        payload.pop("Number of sensors", None)
        set_progress(0)

        # ---- SEED per run riproducibili ----
        if "seed" in payload:
            seed_val = int(payload["seed"])
            np.random.seed(seed_val)
            random.seed(seed_val)   # <--- NEW: rende deterministico anche il meteo

        # ---- PULIZIA UI IMMEDIATA (niente “fantasmi”) ----
        dispersion_placeholder.empty()
        wind_rose_placeholder.empty()
        map_section.empty()
        sensors_placeholder.write("No data available.")
        nps_placeholder.write("N/A")
        source_placeholder.write("N/A")
        metadata_placeholder.empty()

        # --- Binary map generation (cached)
        status_text.text("Binary map generation...")

        try:
            data = cached_binary_map(payload)
        except requests.HTTPError as e:
            st.error(f"Error in binary map generation: {e}")
            return None

        if data.get("status_code") != "success":
            st.error("Error in binary map generation.")
            return None

        set_progress(20)

        
        binary_map = np.array(data.get("map"), dtype=np.float32)
        metadata = data.get("metadata", {})
        free_cells = np.argwhere(binary_map == 1)
        building_cells = np.sum(binary_map == 0)

        with metadata_section:
            metadata_placeholder.markdown(
                f"**Griglia**: {metadata.get('grid_size', 'N/A')}×{metadata.get('grid_size', 'N/A')}\n"
                f"**Edifici totali**: {metadata.get('total_buildings', 'N/A')}\n"
                f"**Celle edifici**: {int(np.sum(building_cells)) if isinstance(building_cells, np.ndarray) else building_cells:,}\n"
                f"**Celle libere**: {int(np.sum(free_cells)) if isinstance(free_cells, np.ndarray) else free_cells:,}\n"
                f"**CRS**: {metadata.get('crs', 'N/A')}\n"
                f"**Risoluzione**: {metadata.get('resolution (m)', 'N/A')} m\n"
                f"**Densità edifici**: {float(metadata.get('building_density', np.nan)):.1f}%\n"
                f"**Altezza media edifici**: {float(metadata.get('mean_height', np.nan))} m\n"
                f"**Città**: {metadata.get('city', 'N/A')}"
            )

        set_progress(40)
        # --- Meteo condition
        status_text.text("Sample meteo condition...")
        sensor_air = SensorAir(sensor_id=00, x=0.0, y=0.0, z=2.0)
        wind_speed, wind_type, stability_type, stability_value, humidify, dry_size, RH = sensor_air.sample_meteorology()

        if weather_section is not None:
            weather_placeholder.markdown(
                f"💨 **Wind speed (m/s):** {wind_speed}  \n"
                f"💨 **Wind type:** {wind_type}  \n"
                f"📈 **Stability:** {stability_type}  \n"
                f"♒︎ **Relative Humidity (%):** {RH}"
            ) 

        # --- Sensor substance
        status_text.text("Air sampling...")
        sensors_substance = []
    
        for i in range(n_sensors):
            x, y = random_position(free_cells)
            sensor_substance = SensorSubstance(i, x=x, y=y, z=2.0,
                                            noise_level=round(np.random.uniform(0.0, 0.0005), 4))
            sensors_substance.append(sensor_substance)

        plot_binary_map(binary_map, metadata['bounds'], map_section, sensors_substance)

        mass_spectrum = []
        for sensor in sensors_substance:
            out = sensor.run_sensor(wind_speed, stability_value, RH, wind_type)
            # Se il sensore è in fault, salta
            if out.get("is_fault", False):
                continue
            spectra = out.get("mass_spectra", [])
            # filtra eventuali NaN
            spectra = [rec for rec in spectra if rec is not None and not np.isnan(rec).any()]
            mass_spectrum.extend(spectra)


        print(f"1->{type(mass_spectrum)}") # list
        print(f"2->{type(mass_spectrum[0])}") # numpy.ndarray
        
        if sensors_section is not None:
            sensor_info = [{"ID": s.id, "x": s.x, "y": s.y, "Status": "Operating" if not s.is_fault else "Faulty",}
                        for s in sensors_substance]
            sensors_placeholder.table(sensor_info)

        set_progress(60)


        # --- NPS classification
        status_text.text("NPS classification...")
        substance_nps = []

        if mass_spectrum:
            spectra_json = [m.tolist() for m in mass_spectrum]
            print(f"spectra_json: {type(spectra_json)}")
            response_dnn = requests.post(f"{CLASSIFIER_URL}/predict_dnn", json={"spectra": spectra_json})

            if response_dnn.status_code == 200:
                predictions = response_dnn.json().get("predictions", [])

                # Enum disponibili (es. FENTANYL, TRYPTAMINES, CANNABINOIDS, PHENETHYLAMINES, PIPERAZINES, ...)
                # Enum disponibili
                nps_classes = [e.name for e in NPS]

                # mappa "elastica" con molteplici alias
                LABEL_MAP = {
                    "fentanyl": "FENTANYL",
                    "fentanyl analogue": "FENTANYL",
                    "fentanyl analogues": "FENTANYL",
                    "fentanyl analogs": "FENTANYL",

                    "tryptamine": "TRYPTAMINES",
                    "tryptamines": "TRYPTAMINES",
                    "tryptamine analogue": "TRYPTAMINES",
                    "tryptamine analogues": "TRYPTAMINES",
                    "tryptamine analogs": "TRYPTAMINES",

                    "cannabinoid": "CANNABINOIDS",
                    "cannabinoids": "CANNABINOIDS",
                    "cannabinoid analogue": "CANNABINOIDS",
                    "cannabinoid analogues": "CANNABINOIDS",
                    "cannabinoid analogs": "CANNABINOIDS",

                    "phenethylamine": "PHENETHYLAMINES",
                    "phenethylamines": "PHENETHYLAMINES",
                    "phenethylamine analogue": "PHENETHYLAMINES",
                    "phenethylamine analogues": "PHENETHYLAMINES",
                    "phenethylamine analogs": "PHENETHYLAMINES",

                    "piperazine": "PIPERAZINES",
                    "piperazines": "PIPERAZINES",
                    "piperazine analogue": "PIPERAZINES",
                    "piperazine analogues": "PIPERAZINES",
                    "piperazine analogs": "PIPERAZINES",
                }

                def normalize_label(label: str) -> str:
                    key = (label or "").strip().lower()
                    key = key.replace("-", " ").replace("_", " ")
                    key = " ".join(key.split())  # collapse spazi multipli
                    # rimuovi suffissi banali
                    for suff in [" class", " classes", " family", " families", " group", " groups"]:
                        if key.endswith(suff):
                            key = key[: -len(suff)]
                            break
                    return LABEL_MAP.get(key, key.upper().replace(" ", "_"))

                mapped = [normalize_label(p) for p in predictions]
                substance_nps = [m for m in mapped if m in nps_classes]

            else:
                st.error(f"Errore API {response_dnn.status_code}")


            # --- DEBUG (opzionale): mostra cosa ha previsto il classificatore ---
            if 'debug_clf' in globals() or 'debug_clf' in locals():
                try:
                    if debug_clf:
                        st.sidebar.markdown("---")
                        st.sidebar.markdown("**🔎 Classifier debug**")
                        st.sidebar.write(f"Spectra sent: {len(mass_spectrum)}")
                        st.sidebar.write(f"Raw predictions (first 10): {predictions[:10] if 'predictions' in locals() else 'N/A'}")
                        if substance_nps:
                            from collections import Counter as Cnt
                            st.sidebar.write("Top labels (NPS-filtered):")
                            top_counts = Cnt(substance_nps).most_common(5)
                            for lbl, cnt in top_counts:
                                st.sidebar.write(f"- {lbl}: {cnt}")
                        else:
                            st.sidebar.write("No labels matched enum NPS.")
                except Exception as _e:
                    st.sidebar.write(f"Debug error: {_e}")



        print(type(substance_nps))
        print(len(substance_nps))

        if substance_nps:
            most_common_substance = Counter(substance_nps).most_common(1)[0][0]
            nps_enum = NPS.from_string(most_common_substance)
        else:
            most_common_substance = None
            nps_enum = NPS.OTHER_COMPOUNDS  # fallback sicuro
            print("Nessuna sostanza presente")


        if nps_section is not None:
            if substance_nps:
                nps_placeholder.markdown(most_common_substance)
            else:
                nps_placeholder.warning("No NPS identified.")

        set_progress(80)


        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        x_src, y_src = random_position(free_cells)
        h_src = round(np.random.uniform(1, 10), 2)  # altezza del pennacchio
        Q = round(np.random.uniform(0.0001, 0.01), 4)  # tasso di emissione
        stacks = [(x_src, y_src, Q, h_src)]

        print(stability_value)
        print(wind_speed)
        print(wind_type)

        param_gaussian_model = ModelConfig(
            days=10,
            RH=RH,
            aerosol_type=nps_enum,
            humidify=humidify,
            stability_profile=stability_type,
            stability_value=stability_value,
            wind_type=wind_type,
            wind_speed=wind_speed,
            output=OutputType.PLAN_VIEW,
            stacks=stacks,
            dry_size=dry_size, x_slice=26, y_slice=1,
            dispersion_model=DispersionModelType.PLUME)

        bounds = (payload["min_lon"], payload["min_lat"], payload["max_lon"], payload["max_lat"])

        response_gauss = requests.post(f"{GAUSSIAN_URL}/start_simulation",
            json={"config": param_gaussian_model.to_dict(),
                "bounds": bounds,
                "seed": int(payload.get("seed", 42))})


        print("risposta ottenuta")
        print(f"code: {response_gauss.status_code}")
        print(response_gauss)

        if response_gauss.status_code != 200:
            st.error("Error in Gaussian puff simulation 01.")
            return sensors_substance, substance_nps, None, None, None, metadata

        gauss_data = response_gauss.json()
        x_raw = gauss_data.get("x", [])
        y_raw = gauss_data.get("y", [])
        times_raw = gauss_data.get("times", [])
        wind_dir_raw = gauss_data.get("wind_dir")
        C1_raw = gauss_data.get("concentration", [])

        x = np.array(x_raw)
        y = np.array(y_raw)
        times = np.array(times_raw)
        wind_dir = np.array(wind_dir_raw)
        C1 = np.array(C1_raw)

        # 🔧 rende x,y,times strettamente crescenti e riallinea C1
        C1, x, y, times = coerce_grid_axes(C1, x, y, times)


        print(type(C1))
        print(C1.shape)
        print(type(wind_dir))
        print(wind_dir.shape)
        print(type(wind_speed))
        print(type(x))
        print(x.shape)
        print(type(y))
        print(y.shape)

        status_text.text("Dispersion map generation...")
        plot_plan_view(C1, x, y, dispersion_placeholder)
        status_text.text("Wind rose graph generation...")
        plot_wind_rose(wind_dir, wind_speed, wind_rose_placeholder)

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # --- Localizzazione sorgente
        status_text.text("Source estimation...")

        payload_sensors = []
        for s in sensors_substance:

            if not s.is_fault:

                s.sample_substance(C1, x, y, times)
                # s.sample_substance_synthetic()

                for idx, (t_idx, conc) in enumerate(zip(s.times, s.noisy_concentrations)):
                    if idx >= len(wind_dir):
                        break
                    wd = wind_dir[idx]

                    payload_sensors.append({
                        "sensor_id": s.id,
                        "sensor_is_fault": s.is_fault,
                        "time": t_idx,
                        "conc": conc if not s.is_fault else None,
                        "wind_dir_x": np.cos(np.deg2rad(wd)) if not s.is_fault else None,
                        "wind_dir_y": np.sin(np.deg2rad(wd)) if not s.is_fault else None,
                        "wind_speed": wind_speed if not s.is_fault else None,
                        "wind_type": wind_type.value if not s.is_fault else None,
                    })

        n_sensor_operating = ([s for s in sensors_substance if not s.is_fault]).__len__()

        status_text.text("Start the prediction of the source...")
        response_loc = requests.post(f"{LOCALIZATION_URL}/predict_source_raw", json={
            "payload_sensors": payload_sensors,
            "n_sensor_operating": n_sensor_operating
        })

        if response_loc.status_code != 200:
            st.error("Error in prediction of source.")

        data = response_loc.json()
        x = data["x"]
        y = data["y"]

        if source_section is not None:
            if x is not None and y is not None:
                source_placeholder.markdown(f"Lat: {x}, Long: {y}")
            else:
                source_placeholder.warning("Source not estimated.")

        set_progress(90)


        # --- gaussian plume dispersion (raw simulation) 
        status_text.text("Raw dispersion simulation...")

        stacks = [(x, y, Q, h_src)]

        param_gaussian_model = ModelConfig(
            days=10,
            RH=RH,
            aerosol_type=nps_enum,
            humidify=humidify,
            stability_profile=stability_type,
            stability_value=stability_value,
            wind_type=wind_type,
            wind_speed=wind_speed,
            output=OutputType.PLAN_VIEW,
            stacks=stacks,
            dry_size=dry_size, x_slice=26, y_slice=1,
            dispersion_model=DispersionModelType.PLUME)

        bounds = (payload["min_lon"], payload["min_lat"], payload["max_lon"], payload["max_lat"])

        response_gauss = requests.post(f"{GAUSSIAN_URL}/start_simulation",
            json={"config": param_gaussian_model.to_dict(),
                "bounds": bounds,
                "seed": int(payload.get("seed", 42))})

            
        if response_gauss.status_code != 200:
            st.error("Error in Gaussian puff simulation.")
            return sensors_substance, substance_nps, None, None, None, metadata

        gauss_data = response_gauss.json()
        x_raw = gauss_data.get("x", [])
        y_raw = gauss_data.get("y", [])
        times_raw = gauss_data.get("times", [])
        wind_dir_raw = gauss_data.get("wind_dir")
        C1_raw = gauss_data.get("concentration", [])

        x_grid = np.array(x_raw)
        y_grid = np.array(y_raw)
        times = np.array(times_raw)
        wind_dir = np.array(wind_dir_raw)
        C1 = np.array(C1_raw)

        # 🔧 come sopra, ma sulle variabili *_grid
        C1, x_grid, y_grid, times = coerce_grid_axes(C1, x_grid, y_grid, times)

        status_text.text("Dispersion map generation...")
        plot_plan_view(C1, x_grid, y_grid, dispersion_placeholder)
        status_text.text("Wind rose graph generation...")
        plot_wind_rose(wind_dir, wind_speed, wind_rose_placeholder)




        # --- Dispersion simulation + correction
        status_text.text("Dispersion simulation...")
        response_mcxm = requests.post(f"{CORRECTION_URL}/correct_dispersion",
                                    json={
                                        "wind_speed": wind_speed,
                                        "wind_dir": wind_dir.tolist(),
                                        "concentration_map": C1.tolist(),
                                        "building_map": binary_map.tolist(),
                                        "global_features": None
                                    })

        if response_mcxm.status_code != 200: 
            st.error("Errore nella correzione della dispersione.") 
            return sensors_substance, substance_nps, x, y, C1, metadata
        
        real_dispersion_map = response_mcxm.json().get("predictions", [])
        real_dispersion_map = np.array(real_dispersion_map)
        print(f"mapp finale {type(real_dispersion_map)}")
        print(real_dispersion_map.shape)

        plot_plan_view(real_dispersion_map, x_grid, y_grid, dispersion_placeholder)

        set_progress(100)
        status_text.text("Simulation completed ✅")
        print("END")
        
        metadata['bounds'] = (payload["min_lon"], payload["min_lat"], payload["max_lon"], payload["max_lat"])
        
        st.session_state.simulation_results = {
            "weather": {"wind_speed": wind_speed, "wind_type": wind_type, "stability": stability_type, "RH": RH},
            "sensors": sensors_substance,
            "nps": most_common_substance,
            "source": (x, y),
            "dispersion_map": real_dispersion_map,
            "metadata": metadata  # Ora metadata contiene anche le bounds
        }
    except Exception as e:
        status_text.error("Si è verificato un errore durante la simulazione.")
        st.exception(e)  # mostro stacktrace nell’app
        # non azzero la UI: lascio placeholders e progress dove sono
        return

# ---------------- INTERFACCIA STREAMLIT ---------------- #
if "simulation_results" not in st.session_state:
    st.session_state.simulation_results = {
        "weather": None,
        "sensors": None,
        "nps": None,
        "source": None,
        "dispersion_map": None,
        "metadata": None
    }

st.markdown(
    """
    <div style="
        position: sticky; 
        top: 0; 
        background-color: white; 
        padding: 20px; 
        z-index: 999; 
        font-size: 36px; 
        font-weight: bold;
        text-align: center;
    ">
        💊 PENTION - NPS Source emission identification system
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(
    """
    <style>
    .start-btn > button {
        background-color: #28a745 !important;
        color: white !important;
        font-weight: bold;
        border-radius: 8px;
        width: 100%;
        padding: 0.5em 0;
    }
    .stop-btn > button {
        background-color: #dc3545 !important;
        color: white !important;
        font-weight: bold;
        border-radius: 8px;
        width: 100%;
        padding: 0.5em 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Layout colonne: lato-sinistra, centro (mappa), lato-destra
col_left, col_center, col_right = st.columns([1, 3, 1])

with col_left:
    weather_section = st.container()
    dispersion_section = st.container()
    metadata_section = st.container()

with col_center:
    map_section = st.container()

with col_right:
    nps_section = st.container()
    source_section = st.container()
    wind_rose_section = st.container()

with weather_section:
    st.markdown("**⛅Meteo conditions**")
    weather_placeholder = st.empty()
    weather_placeholder.markdown(
        f"💨 **Wind speed (m/s):** N/A  \n"
        f"💨 **Wind type:** N/A  \n"
        f"📈 **Stability:** N/A  \n"
        f"♒︎ **Relative Humidity (%):** N/A"
    )

with dispersion_section:
    st.markdown("**🗺️ Dispersion map**")
    dispersion_placeholder = st.empty()

sensors_section = st.container()

with metadata_section:
    st.markdown("**🏙️ Info city map**")
    metadata_placeholder = st.empty()

with sensors_section:
    st.markdown("**🛰️ Sensor**")
    sensors_placeholder = st.empty()
    sensors_placeholder.write("No data available.")

with nps_section:
    st.markdown("**🧪 Nps predicted by sensor**")
    nps_placeholder = st.empty()
    nps_placeholder.write("N/A")

with source_section:
    st.markdown("**📍 Source estimated**")
    source_placeholder = st.empty()
    source_placeholder.write("N/A")

with wind_rose_section:
    st.markdown("🧭 **Wind rose**")
    wind_rose_placeholder = st.empty()

# Sidebar input (come già hai)
st.sidebar.header("Insert simulation parameters")
min_lat = st.sidebar.number_input("Min Lat", value=41.89, format="%.5f")
min_lon = st.sidebar.number_input("Min Lon", value=12.48, format="%.5f")
max_lat = st.sidebar.number_input("Max Lat", value=41.91, format="%.5f")
max_lon = st.sidebar.number_input("Max Lon", value=12.50, format="%.5f")
place = st.sidebar.text_input("Place", value="Insert place name")
n_sensors = st.sidebar.slider("Number of sensors", min_value=5, max_value=50, value=10, step=1)
seed = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)
debug_clf = st.sidebar.checkbox("Debug: show classifier output", value=True)


# Progress & status UNA SOLA VOLTA, con valore persistente
progress_bar = st.sidebar.progress(st.session_state.get('progress', 0))
status_text  = st.sidebar.empty()

# Bottoni con flag "run_in_progress"
col1, col2 = st.sidebar.columns(2)

with col1:
    st.markdown('<div class="start-btn">', unsafe_allow_html=True)
    start = st.button(
        "▶ Start",
        key="start_btn",
        disabled=st.session_state['run_in_progress']
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="stop-btn">', unsafe_allow_html=True)
    stop = st.button(
        "⏹ Stop",
        key="stop_btn",
        disabled=not st.session_state['run_in_progress']
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- LOGICA DI CONTROLLO PRINCIPALE ---------------- #

# Azione per il pulsante START
if start:
    # 1. Imposta lo stato per indicare che la simulazione è in corso
    st.session_state['run_in_progress'] = True
    # 2. Pulisci TUTTI i risultati della simulazione precedente
    st.session_state.simulation_results = {
        "weather": None, "sensors": None, "nps": None,
        "source": None, "dispersion_map": None, "metadata": None
    }
    # 3. Forza un ricaricamento IMMEDIATO della pagina.
    # Questo aggiornerà i bottoni (disabilitando "Start") PRIMA di iniziare i calcoli pesanti.
    st.rerun()

# Azione per il pulsante STOP
if stop:
    # Resetta tutto allo stato iniziale
    st.session_state['run_in_progress'] = False
    st.session_state.simulation_results = {
        "weather": None, "sensors": None, "nps": None,
        "source": None, "dispersion_map": None, "metadata": None
    }
    st.session_state['progress'] = 0
    st.rerun()

# --- ESECUZIONE DELLA SIMULAZIONE E VISUALIZZAZIONE DEI RISULTATI ---

if st.session_state['run_in_progress']:
    # Esegui la simulazione SOLO SE è stata richiesta E non abbiamo ancora i risultati finali.
    # Questo è il cuore della soluzione: evita di rieseguire tutto ad ogni interazione.
    status_text.info("Simulation in progress...")
    payload = {
    "min_lon": min_lon, "min_lat": min_lat,
    "max_lon": max_lon, "max_lat": max_lat,
    "grid_size": 500, "place": place,
    "Number of sensors": n_sensors,
    "seed": int(seed)
    }

    run_application(payload)
    # Una volta finita la simulazione, aggiorna lo stato e ricarica per mostrare i risultati
    st.session_state['run_in_progress'] = False

# Questo blocco viene eseguito sempre, sia prima, durante, che dopo la simulazione.
# Il suo compito è solo mostrare i dati che trova in st.session_state.
results = st.session_state.simulation_results

# Pulisce i placeholder se non ci sono dati (es. dopo aver premuto Stop)
if not results.get("weather"):
    weather_placeholder.markdown(
        f"💨 **Wind speed (m/s):** N/A  \n"
        f"💨 **Wind type:** N/A  \n"
        f"📈 **Stability:** N/A  \n"
        f"♒︎ **Relative Humidity (%):** N/A"
    )
    sensors_placeholder.write("No data available.")
    nps_placeholder.write("N/A")
    source_placeholder.write("N/A")
    wind_rose_placeholder.empty()
    dispersion_placeholder.empty()
    map_section.empty()
    metadata_placeholder.empty()

# Mostra i risultati se sono disponibili
if results.get("weather"):
    weather_placeholder.markdown(
        f"💨 **Wind speed (m/s):** {results['weather']['wind_speed']}  \n"
        f"💨 **Wind type:** {results['weather']['wind_type']}  \n"
        f"📈 **Stability:** {results['weather']['stability']}  \n"
        f"♒︎ **Relative Humidity (%):** {results['weather']['RH']}"
    )

if results.get("sensors"):
    sensor_info = [{"ID": s.id, "x": s.x, "y": s.y, "Status": "Operating" if not s.is_fault else "Faulty"}
                   for s in results["sensors"]]
    sensors_placeholder.table(sensor_info)

if results.get("nps"):
    nps_placeholder.success(results["nps"])
elif results.get("dispersion_map") is not None:
    nps_placeholder.warning("No NPS identified.")

if results.get("source"):
    origin_x, origin_y = results["source"]
    if origin_x is not None:
        source_placeholder.success(f"Coord X: {origin_x:.2f}, Coord Y: {origin_y:.2f}")
    else:
        source_placeholder.warning("Source not estimated.")

if results.get("dispersion_map") is not None and results.get("metadata"):
    with map_section:
        st.subheader("🗺️ Dispersion Map (Final Corrected)")
        bounds = results["metadata"].get("bounds")
        if bounds:
            min_lon, min_lat, max_lon, max_lat = bounds
            m = plot_dispersion_on_map(min_lat, min_lon, max_lat, max_lon,
                                       results["sensors"], results["dispersion_map"],
                                       results["source"][0], results["source"][1])
            st_folium(m, width=700, height=500)