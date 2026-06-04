"""
inference.py
Refactor de la utilidad de inferencia para ser importable desde el dashboard.

Funciones principales:
 - load_resources(base_dir): carga scaler y modelo entrenado
 - prepare_windows_from_csv(csv_path, seq_len): construye X,y y timestamps desde CSV de features
 - predict_windows(model, scaler, X, target_col): predice y devuelve predicciones en unidades reales
 - evaluate(preds, y_true): métricas básicas

También ofrece CLI simple para ejecutar un segmento de prueba.
"""

import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt


BASE_DIR = r"C:\CODE\python\Solar_Irradiance_CU"
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

SCALER_FILE = os.path.join(DATA_DIR, "scaler_final.pkl")
MODEL_FILE = os.path.join(MODEL_DIR, "lstm_model_final.pth")
FEATURES_CSV = os.path.join(DATA_DIR, "nasa_power_features_final.csv")


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def load_resources(base_dir: str = None):
    """Carga scaler y modelo entrenado usando nombres canónicos del pipeline."""
    base = BASE_DIR if base_dir is None else base_dir
    data_dir = os.path.join(base, "data")
    model_dir = os.path.join(base, "models")

    scaler_path = os.path.join(data_dir, "scaler_final.pkl")
    model_path = os.path.join(model_dir, "lstm_model_final.pth")

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler no encontrado: {scaler_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")

    scaler = joblib.load(scaler_path)

    # dummy to read feature count
    df_head = pd.read_csv(os.path.join(data_dir, "nasa_power_features_final.csv"), nrows=1, parse_dates=["datetime"]).drop(columns=["datetime"], errors="ignore")
    input_dim = df_head.shape[1]

    model = LSTMModel(input_dim=input_dim)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    return model, scaler, device


def prepare_windows_from_csv(csv_path: str = FEATURES_CSV, seq_len: int = 24) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Lee CSV de features (sin escalar) y construye X,y y timestamps de predicción.

    Devuelve:
      X: np.array shape (N_windows, seq_len, n_features)
      y: np.array shape (N_windows,) valores reales de GHI (unscaled)
      timestamps: DatetimeIndex de longitud N_windows correspondiento a cada y (fila que se predice)
    """
    df = pd.read_csv(csv_path, parse_dates=["datetime"], index_col="datetime")

    # asegúrate de que la columna objetivo existe
    target_col = "ALLSKY_SFC_SW_DWN"
    if target_col not in df.columns:
        raise KeyError(f"Columna objetivo no encontrada en {csv_path}: {target_col}")

    values = df.values.astype(np.float32)
    n_rows, n_features = values.shape
    n_windows = n_rows - seq_len

    X = np.zeros((n_windows, seq_len, n_features), dtype=np.float32)
    y = np.zeros((n_windows,), dtype=np.float32)
    for i in range(n_windows):
        X[i] = values[i : i + seq_len]
        y[i] = values[i + seq_len, df.columns.get_loc(target_col)]

    timestamps = df.index[seq_len:]
    return X, y, timestamps


def predict_windows(model, scaler, device, X: np.ndarray, target_col: str = "ALLSKY_SFC_SW_DWN") -> np.ndarray:
    """Predice sobre X (raw values). Escala internamente, ejecuta el modelo y devuelve predicciones en unidades reales."""
    n_windows, seq_len, n_features = X.shape

    # Escalar: aplicar scaler.transform a cada fila de cada ventana
    flat = X.reshape(-1, n_features)  # (n_windows*seq_len, n_features)
    flat_scaled = scaler.transform(flat)
    X_scaled = flat_scaled.reshape(n_windows, seq_len, n_features)

    # Inferencia
    with torch.no_grad():
        inp = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        preds_scaled = model(inp).cpu().numpy().flatten()

    # Desnormalizar predicciones (solo la columna target)
    tmp = np.zeros((len(preds_scaled), scaler.n_features_in_), dtype=np.float32)
    # target index from scaler-training CSV (assumes same columns order)
    df_head = pd.read_csv(os.path.join(DATA_DIR, "nasa_power_features_final.csv"), nrows=1, parse_dates=["datetime"]).drop(columns=["datetime"], errors="ignore")
    target_idx = df_head.columns.get_loc(target_col)
    tmp[:, target_idx] = preds_scaled
    preds_denorm = scaler.inverse_transform(tmp)[:, target_idx]

    return preds_denorm


def evaluate(preds: np.ndarray, y_true: np.ndarray) -> dict:
    rmse = np.sqrt(((preds - y_true) ** 2).mean())
    mae = np.mean(np.abs(preds - y_true))
    mape = (np.mean(np.abs((preds - y_true) / (y_true + 1e-6))) * 100)
    return {"rmse": float(rmse), "mae": float(mae), "mape": float(mape)}


def plot_results(y_true: np.ndarray, preds: np.ndarray, title: str = "Inferencia"):
    plt.figure(figsize=(14, 6))
    plt.plot(y_true, label="Real", linewidth=2)
    plt.plot(preds, label="Predicción", linewidth=2)
    plt.title(title)
    plt.xlabel("Tiempo (muestras)")
    plt.ylabel("Irradiancia (W/m²)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # CLI rápido: predecir las últimas N ventanas
    import argparse

    parser = argparse.ArgumentParser(description="Prueba de inferencia LSTM desde CSV de features")
    parser.add_argument("--last-n", type=int, default=500, help="Número de ventanas finales a predecir")
    parser.add_argument("--seq-len", type=int, default=24)
    args = parser.parse_args()

    model, scaler, device = load_resources()
    X, y, timestamps = prepare_windows_from_csv(seq_len=args.seq_len)

    n = len(X)
    last_n = min(args.last_n, n)
    X_seg = X[-last_n:]
    y_seg = y[-last_n:]
    ts_seg = timestamps[-last_n:]

    preds = predict_windows(model, scaler, device, X_seg)
    metrics = evaluate(preds, y_seg)
    print("\n=== MÉTRICAS DE INFERENCIA (segmento final) ===")
    print(f"RMSE: {metrics['rmse']:.3f} W/m²")
    print(f"MAE:  {metrics['mae']:.3f} W/m²")
    print(f"MAPE: {metrics['mape']:.2f} %")

    plot_results(y_seg, preds, title="Inferencia — segmento final")
