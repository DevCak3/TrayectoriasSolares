import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    import io
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass
from torch.utils.data import Dataset, DataLoader
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR   = r"C:\CODE\python\Solar_Irradiance_CU"
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

DATA_FILE   = os.path.join(DATA_DIR, "nasa_power_features_final.csv")
SCALER_FILE = os.path.join(DATA_DIR, "scaler_final.pkl")
MODEL_FILE  = os.path.join(MODEL_DIR, "lstm_model_final.pth")

# Horas solares: franja donde la irradiancia tiene valor energético real.
# Horas nocturnas (0–5 y 20–23) son trivialmente cero — si las incluimos
# en la métrica, el modelo puede obtener MAE bajo simplemente prediciendo
# bien el cero nocturno sin aprender nada sobre los picos solares.
SOLAR_HOURS = set(range(6, 20))   # 06:00 → 19:00 inclusive

# ============================================================
# 1. Cargar dataset en escala original
# ============================================================
df = pd.read_csv(DATA_FILE, parse_dates=["datetime"], index_col="datetime")
# --- Forzar fecha final para evitar datos posteriores a 2025-12-31 en test
END_DATE = pd.to_datetime("2025-12-31 23:00:00")
if df.index.max() > END_DATE:
    df = df.loc[:END_DATE]
    print(f"Dataset recortado a {END_DATE} (filas ahora: {len(df)})")

values_raw = df.values.astype(np.float32)

input_dim = df.shape[1]
seq_len   = 24
GHI_IDX   = 0   # columna 0 = ALLSKY_SFC_SW_DWN

# ============================================================
# 2. Split temporal ANTES de escalar
# ============================================================
n         = len(values_raw)
train_end = int(n * 0.8)
val_end   = int(n * 0.9)

train_raw = values_raw[:train_end]
val_raw   = values_raw[train_end:val_end]
test_raw  = values_raw[val_end:]

# Guardamos también los índices datetime de val y test para poder
# filtrar por hora después de colectar las predicciones.
val_index  = df.index[train_end : val_end]
test_index = df.index[val_end:]

print("Split temporal:")
print(f"  Train : {train_end} filas  ({df.index[0]} → {df.index[train_end-1]})")
print(f"  Val   : {val_end-train_end} filas  ({df.index[train_end]} → {df.index[val_end-1]})")
print(f"  Test  : {n-val_end} filas  ({df.index[val_end]} → {df.index[-1]})")

# ============================================================
# 3. Escalar: fit SOLO sobre train
# ============================================================
scaler = MinMaxScaler()
scaler.fit(train_raw)

train_data = scaler.transform(train_raw).astype(np.float32)
val_data   = scaler.transform(val_raw).astype(np.float32)
test_data  = scaler.transform(test_raw).astype(np.float32)

joblib.dump(scaler, SCALER_FILE)
print(f"\nScaler ajustado con train. GHI range: "
      f"{scaler.data_min_[0]:.1f} – {scaler.data_max_[0]:.1f} W/m²")

# ============================================================
# 4. Utilidades de desnormalización
#    Definidas UNA vez aquí, fuera del loop de epochs.
# ============================================================
def denorm_ghi(scaled_col: np.ndarray) -> np.ndarray:
    """
    Desnormaliza solo la columna GHI sin tocar las demás features.
    inverse_transform necesita un array de forma (n, n_features),
    así que llenamos con ceros y extraemos únicamente la columna GHI.
    """
    tmp = np.zeros((len(scaled_col), scaler.n_features_in_), dtype=np.float32)
    tmp[:, GHI_IDX] = scaled_col
    return scaler.inverse_transform(tmp)[:, GHI_IDX]


def solar_mae(preds_wm2: np.ndarray,
              true_wm2:  np.ndarray,
              timestamps: pd.DatetimeIndex) -> float:
    """
    MAE calculado SOLO sobre horas solares (SOLAR_HOURS).

    Por qué es la métrica correcta para early stopping:
    - Las horas nocturnas (GHI ≈ 0) son trivialmente fáciles de predecir.
      Incluirlas infla artificialmente el rendimiento aparente del modelo.
    - Un modelo que predice bien el cero nocturno pero falla en el mediodía
      puede tener MAE global aceptable pero ser inútil para gestión energética.
    - solar_MAE equivale al mIoU de clases de interés en segmentación:
      mide solo donde el problema es difícil y el resultado importa.

    El índice [seq_len:] alinea las predicciones con sus timestamps reales:
    la primera predicción corresponde a la fila seq_len del set (no a la 0),
    porque las primeras seq_len filas son la ventana de contexto de entrada.
    """
    # Construimos Series indexadas para evitar desalineos sutiles entre
    # arrays numpy y boolean masks provenientes de pandas.
    pred_timestamps = timestamps[seq_len:]

    if len(preds_wm2) != len(pred_timestamps) or len(true_wm2) != len(pred_timestamps):
        raise ValueError("Longitudes inconsistentes entre predicciones/verdaderos y timestamps")

    preds_s = pd.Series(preds_wm2, index=pred_timestamps)
    true_s  = pd.Series(true_wm2,  index=pred_timestamps)

    solar_idx = preds_s.index.hour.isin(SOLAR_HOURS)
    if solar_idx.sum() == 0:
        return float(np.mean(np.abs(preds_s.values - true_s.values)))

    return float(np.mean(np.abs(preds_s[solar_idx] - true_s[solar_idx])))


def hourly_mae_breakdown(preds_wm2:  np.ndarray,
                         true_wm2:   np.ndarray,
                         timestamps: pd.DatetimeIndex) -> dict:
    """
    MAE por cada hora del día (0–23). Útil para el reporte final:
    permite ver exactamente en qué horas falla el modelo.
    """
    pred_timestamps = timestamps[seq_len:]

    if len(preds_wm2) != len(pred_timestamps) or len(true_wm2) != len(pred_timestamps):
        raise ValueError("Longitudes inconsistentes entre predicciones/verdaderos y timestamps")

    preds_s = pd.Series(preds_wm2, index=pred_timestamps)
    true_s  = pd.Series(true_wm2,  index=pred_timestamps)

    result = {}
    for h in range(24):
        sel = preds_s.index.hour == h
        if sel.sum() > 0:
            result[h] = float(np.mean(np.abs(preds_s[sel] - true_s[sel])))
        else:
            result[h] = float("nan")
    return result


# ============================================================
# 5. Dataset — ahora propaga el índice temporal
#
#    El cambio clave respecto a la versión anterior:
#    SequenceDataset ya no devuelve solo (x, y).
#    Devuelve (x, y, idx_posicion) donde idx_posicion permite
#    recuperar el timestamp de cada predicción después de inferencia.
#
#    Alternativa más simple: no modificar el Dataset y en su lugar
#    reconstruir el índice temporal fuera del DataLoader, que es
#    lo que hace solar_mae() con timestamps[seq_len:].
#    Usamos esa alternativa — es más limpia y no rompe compatibilidad
#    con el resto del pipeline (Dashboard, forecast_24h).
# ============================================================
class SequenceDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data    = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len][GHI_IDX]
        return torch.tensor(x), torch.tensor(y)

train_ds = SequenceDataset(train_data, seq_len)
val_ds   = SequenceDataset(val_data,   seq_len)
test_ds  = SequenceDataset(test_data,  seq_len)

# val_loader y test_loader sin shuffle — el orden importa para alinear
# predicciones con timestamps.
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False)

# ============================================================
# 6. Modelo LSTM
# ============================================================
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

device = "cuda" if torch.cuda.is_available() else "cpu"
model  = LSTMModel(input_dim=input_dim).to(device)

# ============================================================
# Función de pérdida: HuberLoss con delta calibrado para irradiancia solar
#
# Por qué Huber en lugar de MSE puro:
#   MSE penaliza cuadráticamente — un error de 200 W/m² genera un gradiente
#   400x mayor que uno de 10 W/m². En días de tormenta o transiciones
#   abruptas nublado→despejado, estos outliers dominan el entrenamiento
#   y empujan los pesos en una dirección que no generaliza.
#
# Cómo funciona Huber:
#   Si |error| <= delta  →  se comporta IGUAL que MSE (zona cuadrática).
#   Si |error| >  delta  →  cambia a pérdida lineal (zona robusta).
#   El gradiente se amortigua: el modelo sigue aprendiendo del outlier,
#   pero con mucha menos presión que con MSE puro.
#
# Por qué delta = 50 W/m²:
#   GHI en CU oscila 0–1000 W/m². Un error de 50 W/m² (~5% del pico)
#   es variabilidad legítima (nubosidad parcial, transición amanecer).
#   Un error >50 W/m² es probable outlier (tormenta abrupta, dato raro).
#   delta=50 traza la frontera entre "ruido normal" y "outlier a suavizar".
#
# Alternativa combinada (más robustez, descomentar si se desea):
#   mse_fn   = nn.MSELoss()
#   mae_fn   = nn.L1Loss()
#   criterion = lambda pred, y: 0.7 * mse_fn(pred, y) + 0.3 * mae_fn(pred, y)
#   Útil si el dataset tiene muchos ceros nocturnos que MAE maneja mejor.
#
# IMPORTANTE: delta está en escala NORMALIZADA [0, 1], no en W/m².
#   El scaler transforma GHI de [0, ~1000] a [0, 1].
#   50 W/m² en escala real ≈ 50 / ghi_max en escala normalizada.
#   Calculamos ghi_max del train para ser precisos:
# ============================================================
ghi_max_train = float(scaler.data_max_[GHI_IDX])   # W/m² máximo visto en train
DELTA_WM2     = 50.0                                 # umbral físico deseado
delta_scaled  = DELTA_WM2 / ghi_max_train            # delta en escala [0,1]

criterion = nn.HuberLoss(delta=delta_scaled)

print(f"\nFunción de pérdida: HuberLoss")
print(f"  delta físico  : {DELTA_WM2:.0f} W/m²")
print(f"  GHI max train : {ghi_max_train:.1f} W/m²")
print(f"  delta escalado: {delta_scaled:.4f}  (usado por PyTorch)")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ReduceLROnPlateau: reduce el LR a la mitad si solar_MAE no mejora
# en 5 epochs. Esto le da al modelo una segunda oportunidad antes de
# que el early stopping (patience=10) lo detenga definitivamente.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5
)

# ============================================================
# 7. Bucle de entrenamiento con early stopping sobre solar_MAE
# ============================================================
best_solar_mae = float("inf")
patience       = 10
wait           = 0

print("\nEntrenando modelo final...")
print(f"Criterio de early stopping: solar_MAE (horas {min(SOLAR_HOURS):02d}:00–{max(SOLAR_HOURS):02d}:00)\n")

for epoch in range(200):

    # ---- Entrenamiento ----
    model.train()
    train_losses = []

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x).squeeze()
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    # ---- Validación: colectar predicciones en orden ----
    model.eval()
    val_preds_scaled = []
    val_true_scaled  = []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            pred = model(x).squeeze().cpu().numpy()
            val_preds_scaled.extend(pred)
            val_true_scaled.extend(y.numpy())

    # ---- Desnormalizar ----
    val_preds_wm2 = denorm_ghi(np.array(val_preds_scaled, dtype=np.float32))
    val_true_wm2  = denorm_ghi(np.array(val_true_scaled,  dtype=np.float32))

    # ---- Métricas ----
    train_loss    = np.mean(train_losses)
    val_mae_all   = float(np.mean(np.abs(val_preds_wm2 - val_true_wm2)))
    val_mae_solar = solar_mae(val_preds_wm2, val_true_wm2, val_index)

    current_lr = optimizer.param_groups[0]["lr"]
    print(f"Epoch {epoch+1:3d} | Train MSE: {train_loss:.4f} | "
          f"MAE global: {val_mae_all:.1f} W/m² | "
          f"MAE solar: {val_mae_solar:.1f} W/m² | "
          f"LR: {current_lr:.2e}")

    # El scheduler observa solar_MAE, no val_loss
    scheduler.step(val_mae_solar)

    # ---- Early stopping: criterio = solar_MAE ----
    if val_mae_solar < best_solar_mae:
        best_solar_mae = val_mae_solar
        wait = 0
        torch.save(model.state_dict(), MODEL_FILE)
        print(f"           ✔ Mejor modelo guardado — solar_MAE: {best_solar_mae:.2f} W/m²")
    else:
        wait += 1
        if wait >= patience:
            print(f"\nEarly stopping activado en epoch {epoch+1}.")
            print(f"Mejor solar_MAE alcanzado: {best_solar_mae:.2f} W/m²")
            break

# ============================================================
# 8. Evaluación final en test
# ============================================================
model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
model.eval()

test_preds_scaled = []
test_true_scaled  = []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        pred = model(x).squeeze().cpu().numpy()
        test_preds_scaled.extend(pred)
        test_true_scaled.extend(y.numpy())

test_preds_wm2 = denorm_ghi(np.array(test_preds_scaled, dtype=np.float32))
test_true_wm2  = denorm_ghi(np.array(test_true_scaled,  dtype=np.float32))

# Sanity: construir Series indexadas por timestamp y mostrar primeras filas
pred_series = pd.Series(test_preds_wm2, index=test_index[seq_len:])
true_series = pd.Series(test_true_wm2,  index=test_index[seq_len:])
print("\nPrimeras predicciones vs reales (test):")
print(pd.DataFrame({'pred': pred_series.head(), 'true': true_series.head()}))

# --- Análisis de desfase temporal: probar shifts en rango [-48, 48] horas
shift_maes = {}
max_shift = 48
for k in range(-max_shift, max_shift + 1):
    shifted = pred_series.shift(k)
    # alineamos índices
    common_idx = shifted.dropna().index.intersection(true_series.index)
    if len(common_idx) == 0:
        continue
    m = mean_absolute_error(true_series.loc[common_idx].values,
                            shifted.loc[common_idx].values)
    shift_maes[k] = m

if len(shift_maes) > 0:
    best_k = min(shift_maes, key=shift_maes.get)
    best_mae = shift_maes[best_k]
    print(f"\nMejor shift detectado: {best_k} horas  —  MAE: {best_mae:.2f} W/m²")
    print(f"MAE sin shift (k=0): {shift_maes.get(0, float('nan')):.2f} W/m²")
    # mostrar unas filas alineadas para el mejor shift
    shifted_best = pred_series.shift(best_k).dropna()
    common_best = shifted_best.index.intersection(true_series.index)
    print("\nEjemplo (mejor shift) — primeras filas: ")
    print(pd.DataFrame({'pred_shifted': shifted_best.loc[common_best].head(),
                        'true': true_series.loc[common_best].head()}))
    # Métricas aplicando el mejor shift (sobre índices comunes)
    mae_global_shift = mean_absolute_error(true_series.loc[common_best].values,
                                           shifted_best.loc[common_best].values)
    rmse_global_shift = np.sqrt(mean_squared_error(true_series.loc[common_best].values,
                                                   shifted_best.loc[common_best].values))
    # solar MAE sobre horas solares del índice común
    solar_mask = common_best.hour.isin(SOLAR_HOURS)
    if solar_mask.sum() > 0:
        mae_solar_shift = float(np.mean(np.abs(shifted_best.loc[common_best][solar_mask].values -
                                               true_series.loc[common_best][solar_mask].values)))
    else:
        mae_solar_shift = float('nan')

    print(f"\nMétricas con shift {best_k}: MAE global {mae_global_shift:.2f} W/m² | "
          f"RMSE {rmse_global_shift:.2f} W/m² | MAE solar {mae_solar_shift:.2f} W/m²")
else:
    print("\nNo se pudo evaluar shifts (sin índices comunes).")

mae_global = mean_absolute_error(test_true_wm2, test_preds_wm2)
rmse_global = np.sqrt(mean_squared_error(test_true_wm2, test_preds_wm2))
mae_solar   = solar_mae(test_preds_wm2, test_true_wm2, test_index)

print("\n=== MÉTRICAS FINALES EN TEST (W/m², escala real) ===")
print(f"MAE  global : {mae_global:.2f} W/m²")
print(f"RMSE global : {rmse_global:.2f} W/m²")
print(f"MAE  solar  : {mae_solar:.2f} W/m²   ← métrica principal")

# ---- Desglose por hora ----
breakdown = hourly_mae_breakdown(test_preds_wm2, test_true_wm2, test_index)
print("\nMAE por franja horaria (W/m²):")
print(f"  {'Hora':<6}  {'MAE':>8}  {'Franja'}")
print(f"  {'----':<6}  {'---':>8}  {'------'}")
for h in range(24):
    mae_h  = breakdown[h]
    franja = "◀ solar" if h in SOLAR_HOURS else "  nocturna"
    if not np.isnan(mae_h):
        print(f"  {h:02d}:00   {mae_h:>7.1f}  {franja}")

print(f"\n✔ Modelo final guardado en:\n{MODEL_FILE}")
