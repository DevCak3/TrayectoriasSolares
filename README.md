# ☀️ Tonatij.app: Predicción Solar y Energética en Ciudad Universitaria (UNAM)

**Tonatij.app** es un framework de configuración de aprendizaje profundo geoespacial diseñado para predecir la **irradiancia solar (GHI)** y la **potencia fotovoltaica (FV)** en el área de Ciudad Universitaria, CDMX. Utiliza una arquitectura de red neuronal **LSTM (Long Short-Term Memory)** entrenada con datos históricos de la NASA y datos meteorológicos en tiempo real.

## 🚀 Características Principales
- **Pipeline de Datos Automatizado:** Descarga y procesa datos horarios de la **NASA POWER API** (periodo 2015-2024).
- **Enriquecimiento de Características:**
  - **Variables Astronómicas:** Cálculo de elevación solar, azimut y factor de distancia Tierra-Sol.
  - **Nubosidad:** Integración de datos de cobertura nubosa (baja, media, alta) desde la API de **Open-Meteo**.
- **Modelo de Deep Learning:** Red LSTM optimizada para series temporales con escalado Min-Max y variables de retardo (*lags*) de hasta 24 horas.
- **Conversión Energética:** Modelo físico para estimar la potencia eléctrica generada (W) a partir de la irradiancia predicha, considerando temperatura y eficiencia de los módulos.
- **Dashboard Interactivo:** Interfaz en **Streamlit** para visualizar predicciones de las próximas 24 horas y exportar resultados en CSV.

## 🛠️ Estructura del Proyecto
El proyecto sigue una estructura organizada para facilitar la reproducibilidad:
- `data/`: Contiene los datasets procesados, el escalador (`scaler_final.pkl`) y los arrays para entrenamiento.
- `models/`: Almacena los pesos del modelo entrenado (`lstm_model_final.pth`).
- `src/`: Scripts de procesamiento, entrenamiento e inferencia.
- `config.py`: Definición de coordenadas del ROI (Ciudad Universitaria), hiperparámetros y rutas de directorios.

## 📦 Requisitos e Instalación
El sistema requiere **Python 3.x** y las siguientes librerías principales:
- `torch`, `pandas`, `numpy`, `scikit-learn`, `streamlit`, `joblib`, `requests` y `matplotlib`.

Para verificar la compatibilidad de PyTorch y CUDA, puedes ejecutar el script `check_torch.py` incluido en las fuentes.

## ⚙️ Flujo de Trabajo (Pipeline)

1.  **Obtención de Datos:** Ejecuta el módulo de descarga para obtener datos históricos de irradiancia, temperatura, humedad y viento.
2.  **Ingeniería de Características:** 
    - Generación de variables astronómicas basadas en la latitud (19.332) y longitud (-99.186) de CU.
    - Incorporación de nubosidad horaria.
    - Creación de *lags* temporales (1h, 2h, 3h, 6h, 12h, 24h) y normalización.
3.  **Entrenamiento:** El modelo se entrena con una división de 80% entrenamiento, 10% validación y 10% prueba, utilizando pérdida MSE y optimizador Adam.
4.  **Despliegue:** Inicia el dashboard con Streamlit para realizar predicciones en tiempo real:
    ```bash
    streamlit run src/dashboard/Dashboard.py
    ```

## 📊 Métricas de Rendimiento
El modelo es evaluado mediante las métricas:
- **MAE** (Error Absoluto Medio)
- **RMSE** (Raíz del Error Cuadrático Medio)
- **MAPE** (Error Porcentual Absoluto Medio).

## 📡 Fuentes de Datos
- **NASA POWER:** Datos meteorológicos y de irradiancia solar.
- **Open-Meteo:** Datos de nubosidad histórica y pronosticada.

---
*Este proyecto fue desarrollado como una solución de Deep Learning para la gestión energética en Ciudad Universitaria.*