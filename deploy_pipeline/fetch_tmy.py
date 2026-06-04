import os
import requests
import pandas as pd
from datetime import datetime
from io import StringIO

# Try to import coordinates from the project script; fallback to CU coords
try:
    from src.Trayectorias_CU_CDMX import LATITUD, LONGITUD
    lat = LATITUD
    lon = -abs(LONGITUD)
except Exception:
    lat = 19.32
    lon = -99.18

BASE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUT_PATH = os.path.join(DATA_DIR, 'nasa_power_hourly_CU_clean.csv')

os.makedirs(DATA_DIR, exist_ok=True)

# NASA POWER hourly point API (request JSON for robust parsing)
API = 'https://power.larc.nasa.gov/api/temporal/hourly/point'
params = {
    'parameters': 'ALLSKY_SFC_SW_DWN',
    'community': 'RE',
    'longitude': lon,
    'latitude': lat,
    'start': '20200101',
    'end': '20201231',
    'format': 'JSON'
}

print(f'Descargando TMY (JSON) desde NASA POWER para lat={lat}, lon={lon}...')
try:
    r = requests.get(API, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    # Navigate JSON structure
    params_dict = data.get('properties', {}).get('parameter', {})
    if not params_dict:
        raise ValueError('Estructura JSON inesperada: no se encontró properties.parameter')

    ghi_series = None
    # find first matching key
    for key in ['ALLSKY_SFC_SW_DWN', 'GHI', 'ghi']:
        if key in params_dict:
            ghi_series = params_dict[key]
            break
    if ghi_series is None:
        # If there's only one parameter, take it
        if len(params_dict) == 1:
            ghi_series = list(params_dict.values())[0]
        else:
            raise ValueError('No se encontró parámetro GHI en la respuesta JSON')

    # ghi_series is a dict mapping YYYYMMDDHH -> value
    rows = []
    for ts, val in ghi_series.items():
        # parse timestamp like '2020010100' or '2020-01-01 00:00'
        try:
            dt = datetime.strptime(ts, '%Y%m%d%H')
        except Exception:
            try:
                dt = pd.to_datetime(ts)
            except Exception:
                continue
        rows.append({'datetime': dt, 'GHI': val})

    df = pd.DataFrame(rows).set_index('datetime').sort_index()
    df.to_csv(OUT_PATH)
    print('TMY (JSON) guardado en:', OUT_PATH)
except Exception as e_json:
    print('Fallo JSON, intentando fallback CSV parseable:', e_json)
    # Fallback: request CSV and parse skipping comment lines
    try:
        params_csv = params.copy()
        params_csv['format'] = 'CSV'
        r2 = requests.get(API, params=params_csv, timeout=30)
        r2.raise_for_status()
        text = r2.text
        # Remove comment lines that start with '#'
        lines = [L for L in text.splitlines() if not L.strip().startswith('#')]
        csv_text = '\n'.join(lines)
        df2 = pd.read_csv(StringIO(csv_text))
        # Normalize GHI column
        for col in ['ALLSKY_SFC_SW_DWN', 'GHI', 'ghi']:
            if col in df2.columns:
                df2 = df2.rename(columns={col: 'GHI'})
                break
        # Try to construct datetime index
        if 'YYYYMMDDHH' in df2.columns:
            df2['datetime'] = pd.to_datetime(df2['YYYYMMDDHH'], format='%Y%m%d%H')
            df2 = df2.set_index('datetime')
        elif 'Year' in df2.columns and 'Month' in df2.columns and 'Day' in df2.columns and 'Hour' in df2.columns:
            df2['datetime'] = pd.to_datetime(df2[['Year','Month','Day','Hour']])
            df2 = df2.set_index('datetime')
        df2.to_csv(OUT_PATH)
        print('TMY (CSV fallback) guardado en:', OUT_PATH)
    except Exception as e_csv:
        print('No se pudo descargar/parsear TMY automáticamente:', e_csv)
        print('Coloca manualmente un archivo TMY en', OUT_PATH)
