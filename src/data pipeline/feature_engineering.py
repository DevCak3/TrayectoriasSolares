import os
import sys
from datetime import datetime

import pandas as pd
import numpy as np
import requests

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_DIR = r"C:\CODE\python\Solar_Irradiance_CU"
DATA_DIR = os.path.join(BASE_DIR, "data")

INPUT_FILE = os.path.join(DATA_DIR, "nasa_power_clean.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "nasa_power_features_final.csv")

# Coordenadas para variables astronómicas y Open-Meteo
LAT = 19.332
LON = -99.186

print("Cargando dataset limpio...")
df = pd.read_csv(INPUT_FILE, parse_dates=["datetime"], index_col="datetime")

print("Generando variables temporales...")
df["hour"] = df.index.hour
df["dayofyear"] = df.index.dayofyear
df["month"] = df.index.month
df["weekday"] = df.index.weekday

print("Generando lags...")
target = "ALLSKY_SFC_SW_DWN"
lags = [1, 2, 3, 6, 12, 24]
for lag in lags:
    df[f"GHI_lag_{lag}h"] = df[target].shift(lag)

# Eliminar filas incompletas generadas por los lags
df = df.dropna()

print("Calculando variables astronómicas...")
phi = np.deg2rad(LAT)
n = df.index.dayofyear.values
B = 2 * np.pi * (n - 1) / 365.0
E0 = (
    1.00011
    + 0.034221 * np.cos(B)
    + 0.00128 * np.sin(B)
    + 0.000719 * np.cos(2 * B)
    + 0.000077 * np.sin(2 * B)
)
df["earth_sun_distance_factor"] = E0

delta = np.deg2rad(23.45) * np.sin(2 * np.pi * (284 + n) / 365.0)
hours = df.index.hour.values
H = np.deg2rad(15 * (hours - 12))
sin_alpha = np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.cos(H)
sin_alpha = np.clip(sin_alpha, -1, 1)
alpha = np.arcsin(sin_alpha)
df["solar_elevation_deg"] = np.rad2deg(alpha)
cos_az = (np.sin(delta) - np.sin(phi) * np.sin(alpha)) / (np.cos(phi) * np.cos(alpha) + 1e-8)
cos_az = np.clip(cos_az, -1, 1)
A = np.arccos(cos_az)
A = np.where(H > 0, 2 * np.pi - A, A)
df["solar_azimuth_deg"] = np.rad2deg(A)

print("Descargando nubosidad desde Open-Meteo...")
start_date = df.index.min().strftime("%Y-%m-%d")
# Forzar fecha final a 2025-12-31 si el dataset contiene timestamps posteriores
end_dt = df.index.max()
FIXED_END = pd.Timestamp("2025-12-31 23:00:00")
if end_dt > FIXED_END:
    print(
        f"Fecha final en datos {end_dt.date()} supera {FIXED_END.date()}; se aplicará {FIXED_END.date()} como tope."
    )
    end_dt = FIXED_END
end_date = end_dt.strftime("%Y-%m-%d")

url = (
    "https://archive-api.open-meteo.com/v1/archive?"
    f"latitude={LAT}&longitude={LON}"
    f"&start_date={start_date}&end_date={end_date}"
    "&hourly=cloudcover,cloudcover_low,cloudcover_mid,cloudcover_high"
    "&timezone=auto"
)
resp = requests.get(url, timeout=60)
data = resp.json()
if "hourly" not in data:
    print("❌ Error en la respuesta de Open-Meteo:")
    print(data)
    raise SystemExit("No se pudo obtener nubosidad horaria.")

cloud_df = pd.DataFrame(
    {
        "datetime": data["hourly"]["time"],
        "cloudcover": data["hourly"]["cloudcover"],
        "cloudcover_low": data["hourly"]["cloudcover_low"],
        "cloudcover_mid": data["hourly"]["cloudcover_mid"],
        "cloudcover_high": data["hourly"]["cloudcover_high"],
    }
)
cloud_df["datetime"] = pd.to_datetime(cloud_df["datetime"])
cloud_df = cloud_df.set_index("datetime")

print("Uniendo nubosidad al dataset...")
df = df.join(cloud_df, how="left")

print("Seleccionando columnas finales...")
feature_cols = [
    "ALLSKY_SFC_SW_DWN",
    "T2M",
    "RH2M",
    "WS2M",
    "solar_elevation_deg",
    "solar_azimuth_deg",
    "earth_sun_distance_factor",
    "cloudcover",
    "cloudcover_low",
    "cloudcover_mid",
    "cloudcover_high",
    "hour",
    "dayofyear",
    "month",
    "weekday",
]
feature_cols += [f"GHI_lag_{lag}h" for lag in lags]

df_final = df[feature_cols].copy()

df_final.to_csv(OUTPUT_FILE)
print(f"Valores min/max de GHI en el dataset completo (solo referencia):")
print(f"  min = {df_final['ALLSKY_SFC_SW_DWN'].min():.2f} W/m²")
print(f"  max = {df_final['ALLSKY_SFC_SW_DWN'].max():.2f} W/m²")
print(f"\n✔ Dataset final SIN escalar guardado en:\n{OUTPUT_FILE}")
print("  El scaler se ajustará en train_model.py, solo sobre el conjunto de entrenamiento.")
