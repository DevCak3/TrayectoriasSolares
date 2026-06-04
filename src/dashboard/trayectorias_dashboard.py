import streamlit as st
import pandas as pd
import numpy as np
import math

import os
import sys
from datetime import date, time as dt_time
import time

# Try to import utilities from project `src` directory. Prefer app-level sys.path
# setup (app.py adds project/src to sys.path). If that fails, add a relative
# path based on this file's location so the app works both locally and on
# Streamlit Cloud.
try:
    from Trayectorias_CU_CDMX import (
        calcular_trayectoria_horaria,
        declinacion,
        angulo_amanecer,
        altitud_solar,
        azimut_solar,
        angulo_incidencia,
        load_tmy,
        ghi_for_day_hour,
        time_correction_minutes,
    )
except Exception:
    # Fallback: add project `src` relative to this file
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_SRC = os.path.abspath(os.path.join(THIS_DIR, ".."))
    if PROJECT_SRC not in sys.path:
        sys.path.insert(0, PROJECT_SRC)
    try:
        from Trayectorias_CU_CDMX import (
            calcular_trayectoria_horaria,
            declinacion,
            angulo_amanecer,
            altitud_solar,
            azimut_solar,
            angulo_incidencia,
            load_tmy,
            ghi_for_day_hour,
            time_correction_minutes,
        )
    except Exception:
        st.error("No se pudieron cargar las utilidades de Trayectorias_CU_CDMX.py. Ejecuta desde el workspace correcto.")
        st.stop()

st.set_page_config(page_title="Trayectorias Solar", layout="wide")
st.title("Calculadora de Trayectorias Solares")
st.markdown("Panel interactivo para explorar altura, azimut, ángulo de incidencia y GHI a lo largo del día para una ubicación dada. Ajusta los parámetros en la barra lateral.")

# Sidebar inputs
with st.sidebar:
    st.header("Parámetros")
    fecha = st.date_input("Día de interés", value=date.today())
    lat = st.number_input("Latitud (°N)", value=19.32, format="%.4f")
    lon = st.number_input("Longitud (°E, oeste negativo)", value=-99.18, format="%.4f")
    inclinacion = st.number_input("Inclinación del panel (°)", value=float(lat), format="%.2f")
    # selector de hora de interés (permite minutos) — colocado antes del disclaimer
    hora_interes = st.time_input("Hora de interés", value=dt_time(12, 0))
    st.markdown("---")
    st.markdown("**Datos TMY:** El dashboard intentará usar datos TMY (NASA POWER) si existen en el proyecto. Los valores TMY son representativos y deben usarse con precaución; verifica su procedencia antes de tomar decisiones de diseño.")
    st.markdown("Si no se encuentra TMY, se usarán valores de respaldo.")
    st.markdown("---")

N = fecha.timetuple().tm_yday

# Cargar TMY opcional (mostrar progreso al descargar)
# Añadir botón para forzar actualización y usar caché para evitar descargas repetidas
if 'tmy_requested' not in st.session_state:
    st.session_state['tmy_requested'] = False

with st.sidebar:
    if st.button('Actualizar/Descargar TMY ahora'):
        st.session_state['tmy_requested'] = True
        # Limpiar caché de datos para forzar nueva descarga
        try:
            st.cache_data.clear()
        except Exception:
            pass
        st.experimental_rerun()


@st.cache_data(ttl=24*3600)
def cached_load_tmy_with_retries(attempts: int = 3, wait_s: float = 1.0):
    """Wrapper cached que intenta descargar/cargar TMY varias veces."""
    for i in range(attempts):
        try:
            df = load_tmy()
            if df is not None:
                return df
        except Exception:
            df = None
        # esperar antes del siguiente intento (pequeño backoff)
        time.sleep(wait_s * (i + 1))
    return None


with st.spinner('Comprobando TMY local y descargando desde NASA POWER si es necesario...'):
    try:
        tmy_df = cached_load_tmy_with_retries()
    except Exception as e:
        tmy_df = None
        st.error(f"Error al intentar cargar/descargar TMY: {e}")

if tmy_df is None:
    st.warning("TMY no encontrado o no disponible; usando valores de respaldo para irradiancia.")
else:
    try:
        st.success(f"TMY cargado correctamente ({len(tmy_df)} filas).")
    except Exception:
        st.success("TMY cargado correctamente.")

# Cálculos: usar la función del script para generar exactamente los puntos
lat_rad = np.radians(lat)

# Obtener trayectoria directamente del script (24 puntos por defecto)
tray = calcular_trayectoria_horaria(N, n_puntos=24, tmy_df=tmy_df)
df = pd.DataFrame(tray)

# Añadir columna GHI consultando TMY usando la hora civil que devuelve la función
if not df.empty:
    if 'hora_solar' in df.columns:
        df['hora_solar'] = df['hora_solar'].astype(float)
    else:
        # Fallback: si no hay hora_solar, calcular aproximada desde h_grados
        corr_min = time_correction_minutes(N, lon, -90.0)
        df['hora_solar'] = (12.0 + df['h_grados'].astype(float)/15.0) - (corr_min/60.0)

    ghi_vals = []
    for hs_val in df['hora_solar']:
        ghi_vals.append(ghi_for_day_hour(tmy_df, N, float(hs_val)) if tmy_df is not None else None)
    df['GHI (W/m2)'] = [round(v,2) if v is not None else None for v in ghi_vals]

# Declinación y correcciones necesarias para cálculos puntuales
delta = declinacion(N)
delta_rad = np.radians(delta)
corr_min = time_correction_minutes(N, lon, -90.0)

# Layout: left controls, right visual
col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("Resumen")
    st.write(f"Fecha: {fecha} (día {N})")
    st.write(f"Latitud: {lat}°N — Longitud: {lon}°E")
    st.write(f"Inclinación panel: {inclinacion}°")
    st.metric("Puntos calculados", len(df))

    if st.button("Descargar CSV"):
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Descargar trayectorias CSV", csv, file_name=f"trayectoria_{fecha}.csv")

with col2:
    st.subheader("Vista tabla")
    # Preparar tabla para mostrar: columna Hora formateada después del índice
    def format_hour(h):
        try:
            h = float(h)
        except Exception:
            return "n/d"
        h_mod = h % 24
        hh = int(math.floor(h_mod))
        mm = int(round((h_mod - hh) * 60))
        if mm == 60:
            hh = (hh + 1) % 24
            mm = 0
        return f"{hh:02d}:{mm:02d}"

    # Insertar columna 'Hora' basada en la hora civil aproximada reportada por el script
    df_display = df.copy()
    # la función del script devuelve 'hora_solar' como hora civil aproximada
    df_display.insert(0, "Hora", df_display["hora_solar"].apply(format_hour))

    # Renombrar columnas a etiquetas legibles con símbolos
    rename_map = {
        "h_grados": "Ángulo horario (ω) [°]",
        "beta_grados": "Altitud (β) [°]",
        "gamma_grados": "Azimut (γ) [°]",
        "theta_grados": "Ángulo incidencia (θ) [°]",
        "GHI (W/m2)": "GHI (W/m²)",
    }
    df_display = df_display.rename(columns=rename_map)

    # Eliminar la columna técnica 'hora_civil' (mostramos 'Hora' legible)
    if 'hora_civil' in df_display.columns:
        df_display = df_display.drop(columns=['hora_civil'])

    # Mostrar la tabla principal (24 horas)
    st.dataframe(df_display)

    # Descarga de la tabla mostrada
    csv = df_display.to_csv(index=False).encode("utf-8")
    st.download_button("Descargar tabla mostrada (CSV)", csv, file_name=f"trayectorias_{fecha}.csv")

    # Panel: detalle para la hora seleccionada (incluye minutos)
    chosen_decimal = hora_interes.hour + hora_interes.minute / 60.0 + hora_interes.second / 3600.0

    # Calcular exactamente para la hora civil elegida (incluye minutos)
    # hora_solar_local = hora_civil + corr_min/60
    hora_solar_local_chosen = chosen_decimal + (corr_min / 60.0)
    h_grados_chosen = (hora_solar_local_chosen - 12.0) * 15.0
    h_rad_chosen = np.radians(h_grados_chosen)

    beta_ch = altitud_solar(lat_rad, delta_rad, h_rad_chosen)
    gamma_ch = azimut_solar(lat_rad, delta_rad, h_rad_chosen, beta_ch)
    theta_ch = angulo_incidencia(lat_rad, np.radians(inclinacion), delta_rad, h_rad_chosen)
    ghi_ch = ghi_for_day_hour(tmy_df, N, chosen_decimal) if tmy_df is not None else None

    sel_row = {
        "Hora civil (h)": f"{hora_interes.strftime('%H:%M')}",
        "Ángulo horario (ω) [°]": round(h_grados_chosen, 4),
        "Altitud (β) [°]": round(beta_ch, 4),
        "Azimut (γ) [°]": (round(gamma_ch, 4) if gamma_ch is not None and not np.isnan(gamma_ch) else None),
        "Ángulo incidencia (θ) [°]": round(theta_ch, 4),
        "GHI (W/m²)": (round(ghi_ch, 2) if ghi_ch is not None else None),
    }

    st.markdown("---")
    st.subheader("Detalle para la hora seleccionada")
    df_sel = pd.DataFrame([sel_row])
    st.table(df_sel)

st.markdown("---")
st.caption("Dashboard creado automáticamente. Ajusta fecha/latitud/longitud para explorar.")
