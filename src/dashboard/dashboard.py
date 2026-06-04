import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import torch
import torch.nn as nn
import joblib
import os
import subprocess
import sys
import time
from datetime import datetime
import streamlit.components.v1 as components

# ============================================================
# Rutas — todas relativas a BASE_DIR para portabilidad
# ============================================================
BASE_DIR    = r"C:\CODE\python\Solar_Irradiance_CU"
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
SRC_DIR     = os.path.join(BASE_DIR, "src")

DATA_FILE        = os.path.join(DATA_DIR, "nasa_power_features_final.csv")
SCALER_FILE      = os.path.join(DATA_DIR, "scaler_final.pkl")
MODEL_FILE       = os.path.join(MODEL_DIR, "lstm_model_final.pth")

# Scripts del pipeline — referenciados por ruta absoluta para que
# subprocess los encuentre independientemente del cwd de Streamlit.
SCRIPT_UPDATE    = os.path.join(SRC_DIR, "data pipeline", "data_fetcher.py")
SCRIPT_FEATURES  = os.path.join(SRC_DIR, "data pipeline", "feature_engineering.py")
SCRIPT_TRAIN     = os.path.join(SRC_DIR, "training", "train_model.py")

# BUG CORREGIDO: una sola asignación, ruta correcta hacia src/inference
INFERENCE_SRC = os.path.normpath(os.path.join(SRC_DIR, "inference"))
if INFERENCE_SRC not in sys.path:
    sys.path.insert(0, INFERENCE_SRC)


target_col = "ALLSKY_SFC_SW_DWN"
GHI_IDX    = 0
device     = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# Modelo LSTM
# ============================================================
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ============================================================
# Carga de datos y modelo — con @st.cache_resource / @st.cache_data
#
# cache_resource: objetos pesados que NO deben recargarse en cada
#   interacción (modelo PyTorch, scaler). Se comparten entre sesiones.
# cache_data: datos que pueden cambiar pero son costosos de leer
#   (CSV grande). Se invalidan explícitamente al actualizar datos.
# ============================================================
@st.cache_resource
def load_model(n_features: int) -> LSTMModel:
    m = LSTMModel(input_dim=n_features).to(device)
    m.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    m.eval()
    return m

@st.cache_data
def load_data():
    """
    Carga el CSV de features y el scaler.
    El CSV sale de feature_engineering.py SIN escalar (corrección Pilar 3).
    El scaler se ajustó solo sobre train — aquí solo hacemos transform.
    """
    df_real  = pd.read_csv(DATA_FILE, parse_dates=["datetime"], index_col="datetime")
    scaler   = joblib.load(SCALER_FILE)

    # Escalar para inferencia (el modelo espera valores normalizados)
    df_scaled = pd.DataFrame(
        scaler.transform(df_real.values),
        index=df_real.index,
        columns=df_real.columns
    )
    return df_scaled, df_real, scaler

# ============================================================
# Forecast de 24 horas
# Separado en su propia función para poder cachearlo con st.cache_data
# y evitar que corra en cada recarga de página.
# ============================================================
@st.cache_data(ttl=3600)   # resultado válido por 1 hora; se recalcula al actualizar datos
def run_forecast(_scaler, _df_scaled_values, _df_real_values, col_names):
    """
    Parámetros como arrays/listas para que st.cache_data pueda hashearlos.
    Los objetos con _ al inicio no se hashean (scaler no es serializable).
    """
    df_scaled = pd.DataFrame(_df_scaled_values, columns=col_names)
    df_real   = pd.DataFrame(_df_real_values,   columns=col_names)

    today = pd.Timestamp.now().normalize()
    hours = [today + pd.Timedelta(hours=i) for i in range(24)]
    preds = []

    window = df_scaled.iloc[-24:].copy()

    # Necesitamos el modelo — lo recuperamos del cache de recursos
    n_features = df_scaled.shape[1]
    mdl = load_model(n_features)

    for current_time in hours:
        x_input = torch.tensor(
            window.values[np.newaxis, :, :], dtype=torch.float32
        ).to(device)

        with torch.no_grad():
            pred_scaled = mdl(x_input).cpu().numpy().flatten()[0]

        # Desnormalizar solo la columna GHI
        tmp = np.zeros((1, _scaler.n_features_in_), dtype=np.float32)
        tmp[0, GHI_IDX] = pred_scaled
        pred_real = float(_scaler.inverse_transform(tmp)[0, GHI_IDX])
        pred_real = max(pred_real, 0.0)
        preds.append(pred_real)

        # Construir fila siguiente para la ventana autoregresiva
        new_row_real = df_real.iloc[-1].copy()
        new_row_real[target_col]    = pred_real
        new_row_real["hour"]        = current_time.hour
        new_row_real["dayofyear"]   = current_time.day_of_year
        new_row_real["month"]       = current_time.month
        new_row_real["weekday"]     = current_time.weekday()

        for lag in [1, 2, 3, 6, 12, 24]:
            new_row_real[f"GHI_lag_{lag}h"] = df_real.iloc[-lag][target_col]

        new_row_scaled_df  = pd.DataFrame([new_row_real], columns=col_names)
        scaled_values      = _scaler.transform(new_row_scaled_df)[0]
        new_row_scaled     = pd.Series(scaled_values, index=col_names)

        window  = pd.concat([window.iloc[1:], new_row_scaled.to_frame().T],
                             ignore_index=True)
        window.columns = col_names
        df_real.loc[len(df_real)] = new_row_real

    return hours, preds

# ============================================================
# Utilidad: lanzar scripts del pipeline en proceso separado
#
# Por qué subprocess.Popen y no subprocess.run:
#   - run() bloquea el hilo de Streamlit hasta que termina.
#     La UI se congela y el usuario no ve ningún progreso.
#   - Popen() lanza el proceso y devuelve control inmediatamente.
#     Streamlit puede mostrar un spinner y el usuario puede
#     seguir viendo la UI mientras el script corre en background.
#
# BUG CORREGIDO: sys.executable en lugar de "python" hardcodeado.
#   En entornos con venv/conda, "python" puede apuntar al Python
#   del sistema (sin las dependencias instaladas), causando ImportError.
#   sys.executable siempre apunta al intérprete activo.
# ============================================================
def run_script(script_path: str, label: str):
    """
    Lanza script_path en un proceso separado y muestra su salida
    en tiempo real dentro de un expander de Streamlit.
    Devuelve True si el proceso terminó con código 0, False si falló.
    """
    if not os.path.exists(script_path):
        st.error(f"Script no encontrado: `{script_path}`")
        return False

    with st.status(f"Ejecutando {label}...", expanded=True) as status:
        proc = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        output_lines = []
        log_area = st.empty()

        for line in proc.stdout:
            output_lines.append(line.rstrip())
            # Mostrar las últimas 20 líneas para no saturar la UI
            log_area.code("\n".join(output_lines[-20:]), language="")

        proc.wait()

        if proc.returncode == 0:
            status.update(label=f"{label} completado", state="complete")
            return True
        else:
            status.update(label=f"{label} falló (código {proc.returncode})", state="error")
            return False

# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="Tonatij.app", page_icon="☀️", layout="wide")

st.title("☀️")
st.title("Tonatij.app")
st.header("Predicción solar y energética · Ciudad Universitaria")
st.caption("Modelo y fuentes: LSTM + NASA POWER + Open-Meteo")

# Valores por defecto (sidebar removida)

# (Barra lateral eliminada: al cargar solo se muestra el título y header)

# ============================================================
# Cuerpo principal
# ============================================================

# Cargar datos
try:
    df_scaled, df_real, scaler = load_data()
except FileNotFoundError as e:
    st.error(f"Archivo no encontrado: `{e.filename}`\n\nEjecuta primero el pipeline desde la línea de comandos.")
    st.stop()

# Cargar modelo
try:
    model = load_model(df_scaled.shape[1])
except FileNotFoundError:
    st.error("Modelo no encontrado. Entrena el modelo ejecutando: src/training/train_model.py")
    st.stop()

# Forecast — corre solo si no hay resultado cacheado o si se invalidó
with st.spinner("Calculando predicción de 24 horas..."):
    hours, future_irr = run_forecast(
        scaler,
        df_scaled.values,
        df_real.values,
        list(df_scaled.columns),
    )

# (PV calculation removed from dashboard — handled externally if needed)

# ---- Tabla de resultados ----
month_names = ["enero","febrero","marzo","abril","mayo","junio",
               "julio","agosto","septiembre","octubre","noviembre","diciembre"]
today    = datetime.now()
date_str = f"{today.day} de {month_names[today.month - 1]}"

# Build output table with only irradiance
df_out = pd.DataFrame({
    "Hora":               [h.strftime("%H:%M") for h in hours],
    "Irradiancia (W/m²)": [round(v, 1) for v in future_irr],
})

# Filtrar filas con irradiancia <= 0 para la tabla, gráfica y descarga
df_out_display = df_out[df_out["Irradiancia (W/m²)"] > 0].reset_index(drop=True)

# ---- Métricas rápidas ----
irr_peak = max(future_irr)
# Métricas: irradiancia pico (con hora) e irradiancia total del día
peak_idx = int(np.argmax(future_irr))
peak_val = future_irr[peak_idx]
peak_hour = hours[peak_idx].strftime('%H:%M')
irr_total = float(np.nansum(future_irr))

col1, col2 = st.columns(2)
col1.metric("Irradiancia pico", f"{peak_val:.0f} W/m² ({peak_hour})")
col2.metric("Irradiancia total (sum)", f"{irr_total:.1f} W·h/m²")

st.subheader(f"Predicción {date_str} · 00:00 → 23:00")

# Tabla fija (completa)
# Inicializar estado de visualización del bloque PV
if "show_pv" not in st.session_state:
    st.session_state["show_pv"] = False

# Botón superior para desplegar la sección de Potencial FV debajo de la gráfica
if st.button("Mostrar Potencial FV"):
    st.session_state["show_pv"] = True

st.table(df_out_display)

# Gráfica fija debajo de la tabla (Altair, tamaño fijo)
chart_df = df_out_display.reset_index(drop=True)
chart = alt.Chart(chart_df).mark_line(point=True).encode(
    x=alt.X('Hora:N', sort=None),
    y=alt.Y('Irradiancia (W/m²):Q')
).properties(height=360)

st.altair_chart(chart, use_container_width=True)

# Descargar CSV
csv_bytes = df_out_display.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Descargar CSV",
    data=csv_bytes,
    file_name=f"forecast_{today.strftime('%Y%m%d')}.csv",
    mime="text/csv",
)

# --- Sección Potencial FV (ancla para desplazamiento) ---
components.html("<div id='pv-section'></div>", height=10)

if st.session_state.get("show_pv", False):
    # Ejecutar pequeño script para desplazar suavemente hasta la sección
    components.html(
        """
        <script>
        const el = document.getElementById('pv-section');
        if(el) { el.scrollIntoView({behavior: 'smooth'}); }
        </script>
        """,
        height=0,
    )

    with st.expander("Potencial Fotovoltaico", expanded=True):
        area = st.number_input("Área del panel (m²)", value=2.57, format="%.3f", step=0.01, key="pv_area")
        eff_pct = st.number_input("Eficiencia (%)", value=21.3, format="%.2f", step=0.1, key="pv_eff")

        eff = float(eff_pct) / 100.0
        # Usar la tabla mostrada (solo horas con irradiancia > 0)
        irr_series = df_out_display["Irradiancia (W/m²)"].astype(float)

        # Energía por hora en Wh: irradiancia (W/m²) * area (m²) * eficiencia * 1 h
        energies_wh = irr_series * float(area) * eff

        df_pv = df_out_display.reset_index(drop=True).copy()
        df_pv["Potencial (Wh)"] = energies_wh.round(1).values
        df_pv["Potencial (kWh)"] = (energies_wh / 1000.0).round(4).values

        total_kwh = float(energies_wh.sum() / 1000.0)
        if len(energies_wh) > 0:
            peak_idx = int(energies_wh.idxmax())
            peak_wh = float(energies_wh.iloc[peak_idx])
            peak_hour = df_pv.iloc[peak_idx]["Hora"]
        else:
            peak_wh = 0.0
            peak_hour = "-"

        st.metric("Potencial diario (kWh)", f"{total_kwh:.3f} kWh")
        st.metric("Pico por hora (Wh)", f"{peak_wh:.0f} Wh ({peak_hour})")

        st.dataframe(df_pv)

        pv_csv = df_pv.to_csv(index=False).encode("utf-8")
        st.download_button("Descargar tabla Potencial FV (CSV)", pv_csv, file_name=f"pv_potential_{today.strftime('%Y%m%d')}.csv")
