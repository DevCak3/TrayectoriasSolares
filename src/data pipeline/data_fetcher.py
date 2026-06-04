import os
import sys
import argparse
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


BASE_DIR = os.path.normpath(r"C:\CODE\python\Solar_Irradiance_CU")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Punto por defecto (Ciudad Universitaria)
LAT = 19.3219
LON = -99.1862


def fetch_nasa_power(start_date: datetime.date, end_date: datetime.date, lat=LAT, lon=LON):
    """Descarga datos horarios de NASA POWER en formato JSON y devuelve DataFrame."""
    base_url = (
        "https://power.larc.nasa.gov/api/temporal/hourly/point?"
        "&parameters=ALLSKY_SFC_SW_DWN,T2M,RH2M,WS2M&community=RE&format=JSON"
    )

    def _request_chunk(s, e):
        url = (
            f"https://power.larc.nasa.gov/api/temporal/hourly/point?start={s.strftime('%Y%m%d')}&"
            f"end={e.strftime('%Y%m%d')}&latitude={lat}&longitude={lon}&community=RE&"
            "parameters=ALLSKY_SFC_SW_DWN,T2M,RH2M,WS2M&format=JSON"
        )
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        return r.json()

    # Intentar petición única primero
    try:
        data = _request_chunk(start_date, end_date)
    except requests.HTTPError:
        # Fallback: dividir en tramos más pequeños y con retroceso de tamaño si fallan
        parts = []
        cur_start = start_date
        # tamaños probados (días) en orden de preferencia
        sizes = [365, 90, 30, 7]

        while cur_start <= end_date:
            fetched = False
            for sz in sizes:
                cur_end = min(cur_start + timedelta(days=sz - 1), end_date)
                try:
                    print(f"Intentando tramo {cur_start} → {cur_end} (size={sz})...")
                    data_part = _request_chunk(cur_start, cur_end)
                    parts.append(data_part)
                    cur_start = cur_end + timedelta(days=1)
                    fetched = True
                    break
                except requests.HTTPError as e:
                    print(f"Tramo {cur_start} → {cur_end} falló (size={sz}), intentando tamaño menor...")
                    continue
            if not fetched:
                raise RuntimeError(f"No se pudo descargar el tramo empezando en {cur_start}; intenta reducir el rango manualmente.")

        # Combinar partes en un único diccionario-like de 'parameter'
        combined = {}
        params = ["ALLSKY_SFC_SW_DWN", "T2M", "RH2M", "WS2M"]
        for p in params:
            combined[p] = {}
        for part in parts:
            recs = part["properties"]["parameter"]
            for p in params:
                combined[p].update(recs[p])

        data = {"properties": {"parameter": combined}}

    if "properties" not in data or "parameter" not in data["properties"]:
        raise RuntimeError("Respuesta inesperada de NASA POWER")

    records = data["properties"]["parameter"]

    # Las claves vienen como 'YYYYMMDDHH' (ej: '2015010100'). Parsearlas con formato fijo
    keys = list(records["ALLSKY_SFC_SW_DWN"].keys())
    # Determinar formato: si todos son digitos y longitud 10 => %Y%m%d%H
    if all(k.isdigit() and len(k) == 10 for k in keys):
        dt_index = pd.to_datetime(keys, format="%Y%m%d%H", errors="coerce")
    else:
        # Fallback más permisivo
        dt_index = pd.to_datetime(keys, errors="coerce")

    if dt_index.isna().any():
        # Intentar limpiar claves no estándar (quitar sufijos)
        cleaned = [k.rstrip('Z') for k in keys]
        dt_index = pd.to_datetime(cleaned, format="%Y%m%d%H", errors="coerce")

    if dt_index.isna().any():
        raise RuntimeError("No se pudieron parsear las marcas de tiempo de NASA POWER")

    df = pd.DataFrame({
        "datetime": dt_index,
        "ALLSKY_SFC_SW_DWN": list(records["ALLSKY_SFC_SW_DWN"].values()),
        "T2M": list(records["T2M"].values()),
        "RH2M": list(records["RH2M"].values()),
        "WS2M": list(records["WS2M"].values()),
    })

    df = df.sort_values("datetime").set_index("datetime")
    return df


def clean_and_save(df: pd.DataFrame, start_date: datetime.date, end_date: datetime.date):
    # Interpolar faltantes si los hay
    if df.isna().sum().sum() > 0:
        print("Rellenando valores faltantes mediante interpolación temporal...")
        df = df.interpolate(method="time")

    out1 = os.path.join(DATA_DIR, "nasa_power_clean.csv")
    out2 = os.path.join(DATA_DIR, "nasa_power_hourly_CU_clean.csv")
    out3 = os.path.join(DATA_DIR, f"nasa_power_hourly_{start_date.strftime('%Y')}_{end_date.strftime('%Y')}.csv")

    df.to_csv(out1)
    df.to_csv(out2)
    df.to_csv(out3)

    print(f"Dataset guardado en:\n  {out1}\n  {out2}\n  {out3}")


def detect_span_days(candidate_files):
    for f in candidate_files:
        if os.path.exists(f):
            try:
                df_prev = pd.read_csv(f, parse_dates=["datetime"], index_col="datetime")
                span = df_prev.index.max().date() - df_prev.index.min().date()
                return span.days
            except Exception:
                continue
    return None


def main():
    parser = argparse.ArgumentParser(description="Data fetcher para NASA POWER (bulk | update)")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_bulk = sub.add_parser("bulk", help="Descarga histórico completo o rango especificado")
    p_bulk.add_argument("--start", help="Fecha inicio YYYYMMDD", default=None)
    p_bulk.add_argument("--end", help="Fecha fin YYYYMMDD (por defecto: 20251231)", default=None)

    p_upd = sub.add_parser("update", help="Descarga la ventana histórica desplazada hasta la fecha más reciente")
    p_upd.add_argument("--days-back", type=int, default=0, help="Desplazar end_date N días atrás (por defecto 0)")

    args = parser.parse_args()

    # Forzar fecha máxima de consulta
    FIXED_END = datetime(2025, 12, 31).date()
    default_end = FIXED_END

    candidate_files = [
        os.path.join(DATA_DIR, "nasa_power_clean.csv"),
        os.path.join(DATA_DIR, "nasa_power_hourly_CU_clean.csv"),
    ]

    if args.mode == "bulk":
        if args.end:
            end_date = datetime.strptime(args.end, "%Y%m%d").date()
        else:
            end_date = default_end
        # no permitir fechas posteriores a FIXED_END
        if end_date > FIXED_END:
            print(f"Fecha de fin solicitada {end_date} supera {FIXED_END}, se aplica {FIXED_END}.")
            end_date = FIXED_END

        if args.start:
            start_date = datetime.strptime(args.start, "%Y%m%d").date()
        else:
            start_date = datetime(2015, 1, 1).date()

    else:  # update
        span_days = detect_span_days(candidate_files)
        if span_days is None:
            span_days = 365 * 10
            print(f"No se detectó dataset previo. Usando ventana por defecto: {span_days} días")

        end_date = default_end - timedelta(days=args.days_back)
        if end_date > FIXED_END:
            end_date = FIXED_END
        start_date = end_date - timedelta(days=span_days)

    print(f"Solicitando datos NASA POWER para: {start_date} → {end_date}")
    df = fetch_nasa_power(start_date, end_date)
    clean_and_save(df, start_date, end_date)


if __name__ == "__main__":
    main()
