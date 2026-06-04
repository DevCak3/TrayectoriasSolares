# config.py
import os

# --- CONFIGURACIÓN GEOSPACIAL ---
# Coordenadas de Ciudad Universitaria provistas por el usuario
ROI_COORDINATES = [
    [-99.1992065934911, 19.337621517608852], # Top-Left / Esquinas del rectángulo
    [-99.17328572557118, 19.306194979956107] # Bottom-Right
]

# --- PARÁMETROS DEL DATASET ---
START_YEAR = 2021
END_YEAR = 2025  # Rango temporal para análisis histórico semanal
RESOLUTION_METERS = 10  # Forzar a escala de Sentinel-2

# --- RUTA DE DIRECTORIOS (Compatibilidad Local/Colab)
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed")

# --- HIPERPARÁMETROS DE DEEP LEARNING (Para Colab)
BATCH_SIZE = 16
LEARNING_RATE = 1e-4  # Evitar nombres cortos como 'lr' según sugerencia docente
EPOCHS = 50
BACKBONE = "resnet34"  # Transfer learning encoder