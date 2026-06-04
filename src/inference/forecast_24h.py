import os
import subprocess
import sys
import numpy as np
import pandas as pd
import torch

# ------------------------------------------------------------
# 0) Ejecutar actualización automática de datos (data_fetcher.py)
# ------------------------------------------------------------
BASE_DIR = r"C:\CODE\python\Solar_Irradiance_CU"
SRC_DIR = os.path.join(BASE_DIR, "src")
UPDATE_SCRIPT = os.path.join(SRC_DIR, "data pipeline", "data_fetcher.py")

print("🔄 Actualizando datos con NASA POWER + Open‑Meteo...")
try:
    subprocess.run([sys.executable, UPDATE_SCRIPT], check=True)
    print("✔ Datos actualizados correctamente.\n")
except Exception as e:
    print(f"! La actualización falló: {e}. Continuando con los datos existentes.")

# ------------------------------------------------------------
# Rutas
# ------------------------------------------------------------
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

DATA_FILE = os.path.join(DATA_DIR, "nasa_power_features_final.csv")
target_col = "ALLSKY_SFC_SW_DWN"

# ------------------------------------------------------------
# 1) Cargar scaler/modelo mediante helpers de inference.py y leer CSV
# ------------------------------------------------------------
INFERENCE_SRC = os.path.normpath(os.path.join(SRC_DIR, "inference"))
if INFERENCE_SRC not in sys.path:
    sys.path.insert(0, INFERENCE_SRC)

try:
    from inference import load_resources
except Exception:
    # Fallback: intentar cargar directamente si el módulo cambia
    from inference import load_resources

model, scaler, device = load_resources(BASE_DIR)

# Leer CSV crudo (sin escalar) y escalar para inferencia
df_real = pd.read_csv(DATA_FILE, parse_dates=["datetime"], index_col="datetime")
df_scaled = pd.DataFrame(
    scaler.transform(df_real.values), index=df_real.index, columns=df_real.columns
)

print("Modelo y datos cargados correctamente.\n")

# ------------------------------------------------------------
# 3) Forecast de 24 horas (00:00 → 23:00)
# ------------------------------------------------------------
def forecast_day(model, scaler, device, df_real, df_scaled, horizon: int = 24):
    df_cols = df_scaled.columns
    today = pd.Timestamp.now().normalize()
    hours = [today + pd.Timedelta(hours=i) for i in range(horizon)]
    preds = []

    window_size = 24
    window = df_scaled.iloc[-window_size:].copy()

    for current_time in hours:
        x_input = torch.tensor(window.values[np.newaxis, :, :], dtype=torch.float32).to(device)
        with torch.no_grad():
            pred_scaled = model(x_input).cpu().numpy().flatten()[0]

        tmp = np.zeros((1, scaler.n_features_in_), dtype=np.float32)
        tmp[0, df_cols.get_loc(target_col)] = pred_scaled
        pred_real = float(scaler.inverse_transform(tmp)[0, df_cols.get_loc(target_col)])
        pred_real = max(pred_real, 0.0)
        preds.append(pred_real)

        new_row_real = df_real.iloc[-1].copy()
        new_row_real[target_col] = pred_real
        new_row_real["hour"] = current_time.hour
        new_row_real["dayofyear"] = current_time.day_of_year
        new_row_real["month"] = current_time.month
        new_row_real["weekday"] = current_time.weekday()

        for lag in [1, 2, 3, 6, 12, 24]:
            new_row_real[f"GHI_lag_{lag}h"] = df_real.iloc[-lag][target_col]

        new_row_scaled_df = pd.DataFrame([new_row_real], columns=df_cols)
        scaled_values = scaler.transform(new_row_scaled_df)[0]
        new_row_scaled = pd.Series(scaled_values, index=df_cols)

        window = pd.concat([window.iloc[1:].reset_index(drop=True), new_row_scaled.to_frame().T.reset_index(drop=True)], ignore_index=True)
        window.columns = df_cols
        df_real.loc[current_time] = new_row_real

    return hours, preds

# ------------------------------------------------------------
# 4) Ejecutar forecasting de 24 horas
# ------------------------------------------------------------
hours, future = forecast_day(model, scaler, device, df_real.copy(), df_scaled.copy(), horizon=24)

print("Predicciones de irradiancia para el día de hoy (W/m²):")
for h, p in zip(hours, future):
    print(f"{h.strftime('%H:%M')} → {p:.2f}")

# ------------------------------------------------------------
# 5) Conversión a potencia FV (opcional, simple PVWatts-like)
# ------------------------------------------------------------
def pv_power_from_irradiance(ghi_values, temp_values, area_total=10.0, eta_ref=0.18, gamma=-0.0045, NOCT=45):
    ghi_values = np.array(ghi_values)
    temp_values = np.array(temp_values)
    T_cell = temp_values + (ghi_values / 800.0) * NOCT
    eta_temp = eta_ref * (1 + gamma * (T_cell - 25.0))
    eta_temp = np.maximum(eta_temp, 0.0)
    P = ghi_values * area_total * eta_temp
    P = np.maximum(P, 0.0)
    return P

if "T2M" in df_real.columns:
    temps = df_real["T2M"].iloc[-24:].values
    pv_power = pv_power_from_irradiance(ghi_values=future, temp_values=temps, area_total=10.0, eta_ref=0.18)
    print("\nPotencia FV estimada (W):")
    for h, p in zip(hours, pv_power):
        print(f"{h.strftime('%H:%M')} → {p:.2f} W")
else:
    print("\nNo se encontró columna 'T2M' para calcular potencia FV. Omitido.")
