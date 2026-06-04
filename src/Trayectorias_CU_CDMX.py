import numpy as np
import math
import csv
import os
import pandas as pd

# =============================================================================
#  TRAYECTORIAS SOLARES — Ciudad Universitaria (CDMX)
#  Proyecto Final: Aprovechamiento de Energía Solar
#  Facultad de Ingeniería, UNAM
# =============================================================================
#
#  Parámetros fijos del proyecto:
#    Latitud  : 19.32° N
#    Longitud : 99.18° O
#    Inclinación óptima del panel (w): 19.32° (igual a la latitud, orientación sur)
#
#  Fechas representativas:
#    Fecha 1 — 21 de marzo  (N=80)  : Equinoccio de primavera
#    Fecha 2 — 22 de septiembre (N=265): Equinoccio de otoño
# =============================================================================

# ---------------------------------------------------------------------------
# Parámetros del proyecto
# ---------------------------------------------------------------------------
LATITUD   = 19.32    # grados Norte
LONGITUD  = 99.18    # grados Oeste (negativo en convención Este positivo)
INCLINACION_PANEL = 19.32   # w  [°] — igual a la latitud, orientación sur

# Constante solar [W/m²]
ISC = 1353.0

# Especificaciones del panel fotovoltaico (LONGi LR5-72HBD-550M / Trina TSM-550)
POTENCIA_PICO  = 550.0   # Wp
AREA_PANEL     = 2.58    # m²
EFICIENCIA     = 0.213   # 21.3 %

# Fechas representativas {nombre: N}
FECHAS = {
    "21_marzo_equinoccio_primavera": 80,
    "22_septiembre_equinoccio_otono": 265,
}

# Horas solares representativas para cálculo detallado
HORAS_REPRESENTATIVAS = {
    "Amanecer": None,   # se calcula como h = hs
    "10:00 am": 30.0,   # ángulo horario = 30° (2 h antes del mediodía)
    "Mediodía": 0.0,    # h = 0°
}


# ---------------------------------------------------------------------------
# Funciones de geometría solar
# ---------------------------------------------------------------------------

def declinacion(N: float) -> float:
    """Declinación solar δ [°] — fórmula de Cooper."""
    return 23.45 * math.cos(math.radians(360 / 365 * (N - 173)))


def equation_of_time_minutes(N: int) -> float:
    """Aproximación de la Ecuación del Tiempo (ET) en minutos.

    Fórmula común: B = 360*(N-81)/364 (grados)
    ET = 9.87*sin(2B) - 7.53*cos(B) - 1.5*sin(B)
    """
    B_deg = 360.0 * (N - 81) / 364.0
    B = math.radians(B_deg)
    et = 9.87 * math.sin(2 * B) - 7.53 * math.cos(B) - 1.5 * math.sin(B)
    return et


def time_correction_minutes(N: int, longitude_local_deg_east: float, longitude_std_deg_east: float = -90.0) -> float:
    """Devuelve la corrección en minutos a aplicar entre Tiempo Solar y Tiempo Estándar.

    Usamos: Tiempo Solar = Tiempo Estándar + 4*(longitude_std - longitude_local) + ET
    con longitudes en grados (este positivo). La función devuelve 4*(long_std - long_local)+ET.
    """
    et = equation_of_time_minutes(N)
    # corrección por longitud (4 minutos por grado)
    corr = 4.0 * (longitude_std_deg_east - longitude_local_deg_east)
    return corr + et


def angulo_amanecer(lat_rad: float, delta_rad: float) -> float:
    """Ángulo horario del amanecer hs [°]."""
    valor = -math.tan(lat_rad) * math.tan(delta_rad)
    valor = max(-1.0, min(1.0, valor))
    return math.degrees(math.acos(valor))


def altitud_solar(lat_rad: float, delta_rad: float, h_rad: float) -> float:
    """Altitud solar β [°]."""
    sin_b = (math.sin(lat_rad) * math.sin(delta_rad) +
             math.cos(lat_rad) * math.cos(delta_rad) * math.cos(h_rad))
    sin_b = max(-1.0, min(1.0, sin_b))
    return math.degrees(math.asin(sin_b))


def azimut_solar(lat_rad: float, delta_rad: float, h_rad: float, beta: float) -> float:
    """Ángulo de azimut γ [°] medido desde el norte."""
    # beta en este código es la altitud solar (grados).
    # Evitar divisiones por valores cercanos a cero al amanecer/ocaso.
    # En lugar de dividir por cos(β_cenital), calculamos las componentes
    # multiplicadas por cos(β_cenital) y usamos atan2:
    #   C = cosβ * cosγ = sinδ·cosL − cosδ·sinL·cosh
    #   S = cosβ * sinγ = cosδ·sinh
    # Entonces γ = atan2(S, C) y no necesitamo dividir por cosβ.
    # Esto proporciona un valor estable cuando β → 0.

    C = (math.sin(delta_rad) * math.cos(lat_rad)
         - math.cos(delta_rad) * math.sin(lat_rad) * math.cos(h_rad))
    S = (math.cos(delta_rad) * math.sin(h_rad))

    # Limitar C a [-1,1] no es necesario antes de atan2, pero mantenemos valores finitos
    gamma_rad = math.atan2(S, C)
    return math.degrees(gamma_rad)


def angulo_incidencia(lat_rad: float, w_rad: float, delta_rad: float, h_rad: float) -> float:
    """
    Ángulo de incidencia θ [°] para panel inclinado orientado al sur.
    Fórmula simplificada (ec. 8 del proyecto):
        cos θ = cos(L-w)·cos δ·cos h + sin(L-w)·sin δ
    """
    lw_rad = lat_rad - w_rad
    cos_theta = (math.cos(lw_rad) * math.cos(delta_rad) * math.cos(h_rad) +
                 math.sin(lw_rad) * math.sin(delta_rad))
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.degrees(math.acos(cos_theta))


def potencia_absorbida(I_incidente: float, cos_theta: float) -> dict:
    """
    Potencia normal y real absorbida por el panel [W].
    Pnormal = I · A · η
    Pabs    = I · A · η · cos θ
    """
    p_normal = I_incidente * AREA_PANEL * EFICIENCIA
    p_abs    = p_normal * cos_theta
    return {"Pnormal_W": round(p_normal, 2), "Pabs_W": round(p_abs, 2)}


def load_tmy(tmy_path: str = None) -> pd.DataFrame:
    """Carga un archivo TMY/horario con columna GHI.

    Busca columnas comunes: 'ALLSKY_SFC_SW_DWN', 'GHI', 'ghi'.
    Si no se encuentra o el archivo no existe, devuelve None.
    """
    # Ruta por defecto relativa al proyecto
    if tmy_path is None:
        base = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
        tmy_path = os.path.join(base, "data", "nasa_power_hourly_CU_clean.csv")

    if not os.path.exists(tmy_path):
        return None

    try:
        df = pd.read_csv(tmy_path, parse_dates=["datetime"], index_col="datetime")
    except Exception:
        try:
            df = pd.read_csv(tmy_path, parse_dates=[0], index_col=0)
        except Exception:
            return None

    # Normalizar nombre de columna de irradiancia
    for col in ["ALLSKY_SFC_SW_DWN", "GHI", "ghi", "ghi_w/m2"]:
        if col in df.columns:
            df = df.rename(columns={col: "GHI"})
            break

    if "GHI" not in df.columns:
        return None

    # Asegurar frecuencia horaria
    df = df.sort_index()
    return df


def ghi_for_day_hour(tmy_df: pd.DataFrame, N: int, hour_decimal: float) -> float:
    """Obtiene GHI para el día del año N y la hora (decimal). Redondea a la hora más cercana.

    Si no hay datos, devuelve None.
    """
    if tmy_df is None:
        return None

    # Redondear al entero más cercano (0-23)
    h_rounded = int(round(hour_decimal)) % 24
    # Buscar filas con mismo día del año y hora
    try:
        doy = tmy_df.index.dayofyear
        mask = (doy == N) & (tmy_df.index.hour == h_rounded)
        sel = tmy_df.loc[mask]
        if not sel.empty:
            # Si hay varias entradas (por ejemplo distintas years), tomar la media
            return float(sel["GHI"].mean())
    except Exception:
        return None

    return None


# ---------------------------------------------------------------------------
# Cálculo principal
# ---------------------------------------------------------------------------

def calcular_para_fecha(nombre_fecha: str, N: int, tmy_df: pd.DataFrame = None) -> dict:
    """Calcula todos los ángulos solares y potencias para una fecha dada."""

    lat_rad   = math.radians(LATITUD)
    w_rad     = math.radians(INCLINACION_PANEL)

    delta     = declinacion(N)
    delta_rad = math.radians(delta)

    hs        = angulo_amanecer(lat_rad, delta_rad)
    duracion  = 2 * hs / 15.0   # horas

    # Altitud máxima (mediodía, h=0)
    beta_max  = 90.0 - abs(LATITUD - delta)

    resultados_horarios = []

    # Longitud local en convención Este positivo (LONGITUD está en grados Oeste positivos)
    longitude_local = -abs(LONGITUD)
    longitude_std = -90.0

    for nombre_hora, h_val in HORAS_REPRESENTATIVAS.items():
        if h_val is None:
            h_grados = hs   # amanecer
        else:
            h_grados = h_val

        h_rad  = math.radians(h_grados)
        beta   = altitud_solar(lat_rad, delta_rad, h_rad)
        gamma  = azimut_solar(lat_rad, delta_rad, h_rad, beta)
        theta  = angulo_incidencia(lat_rad, w_rad, delta_rad, h_rad)
        cos_t  = math.cos(math.radians(theta))

        # Hora solar (local): LST_solar = 12 + ω/15
        hora_solar_local = 12.0 + (h_grados / 15.0)

        # Corrección por longitud y Ecuación del Tiempo (minutos)
        corr_min = time_correction_minutes(N, longitude_local, longitude_std)
        # Tiempo civil (reloj) aproximado = hora_solar_local - corr_min/60
        hora_civil = hora_solar_local - (corr_min / 60.0)

        I_from_tmy = ghi_for_day_hour(tmy_df, N, hora_civil)
        if I_from_tmy is not None:
            I_inc = I_from_tmy
        else:
            # Fallback a valores típicos si no hay TMY
            fallback = {"Amanecer": 200.0, "10:00 am": 800.0, "Mediodía": 1000.0}
            I_inc = fallback.get(nombre_hora, 500.0)
        pots   = potencia_absorbida(I_inc, cos_t)

        # Mostrar hora civil (reloj) en la salida para mayor claridad
        hora_civil = hora_civil

        resultados_horarios.append({
            "Hora civil (aprox.)": f"{hora_civil:.2f} h",
            "h [°]"    : round(h_grados, 3),
            "β [°]"    : round(beta, 2),
            "γ [°]"    : round(gamma, 2) if not math.isnan(gamma) else "n/d",
            "cos θ"    : round(cos_t, 5),
            "θ [°]"    : round(theta, 2),
            "I [W/m²]" : I_inc,
            "Pnormal [W]": pots["Pnormal_W"],
            "Pabs [W]"   : pots["Pabs_W"],
        })

    return {
        "Fecha"              : nombre_fecha,
        "N"                  : N,
        "δ [°]"              : round(delta, 4),
        "hs [°]"             : round(hs, 2),
        "Duración del día [h]": round(duracion, 3),
        "βmax [°]"           : round(beta_max, 2),
        "Inclinación panel [°]": INCLINACION_PANEL,
        "Horarios"           : resultados_horarios,
    }


def calcular_trayectoria_horaria(N: int, n_puntos: int = 12, tmy_df: pd.DataFrame = None) -> list:
    """
    Trayectoria completa (n_puntos a lo largo del día) para graficar o exportar.
    Retorna lista de dicts con h, β, γ, θ.
    """
    lat_rad   = math.radians(LATITUD)
    w_rad     = math.radians(INCLINACION_PANEL)
    delta     = declinacion(N)
    delta_rad = math.radians(delta)
    hs        = angulo_amanecer(lat_rad, delta_rad)

    angulos_h = np.linspace(-hs, hs, n_puntos)
    trayectoria = []

    for h_grados in angulos_h:
        h_rad  = math.radians(h_grados)
        beta   = altitud_solar(lat_rad, delta_rad, h_rad)
        gamma  = azimut_solar(lat_rad, delta_rad, h_rad, beta)
        theta  = angulo_incidencia(lat_rad, w_rad, delta_rad, h_rad)
        # Corregir hora solar a hora civil usando ET y corrección por longitud
        longitude_local = -abs(LONGITUD)
        longitude_std = -90.0
        hora_solar_local = 12.0 + h_grados / 15.0
        corr_min = time_correction_minutes(N, longitude_local, longitude_std)
        hora = hora_solar_local - (corr_min / 60.0)
        # Si tenemos TMY, podríamos anotar GHI por punto (opcional). No lo escribimos
        # en la salida principal (solo ángulos), pero la función podría extenderse.
        trayectoria.append({
            "N": N,
            "hora_solar": round(hora, 4),
            "h_grados"  : round(h_grados, 4),
            "beta_grados": round(beta, 4),
            "gamma_grados": round(gamma, 4) if not math.isnan(gamma) else None,
            "theta_grados": round(theta, 4),
        })

    return trayectoria


# ---------------------------------------------------------------------------
# Exportación a CSV
# ---------------------------------------------------------------------------

def exportar_csv(trayectorias: list, csv_path: str):
    os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else ".", exist_ok=True)
    campos = ["N", "hora_solar", "h_grados", "beta_grados", "gamma_grados", "theta_grados"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(trayectorias)
    print(f"  → CSV guardado en: {csv_path}")


# ---------------------------------------------------------------------------
# Impresión de resultados en consola
# ---------------------------------------------------------------------------

def imprimir_resultados(res: dict):
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  FECHA: {res['Fecha'].replace('_', ' ').upper()}")
    print(f"  N = {res['N']}")
    print(sep)
    print(f"  Latitud            : {LATITUD}° N")
    print(f"  Longitud           : {LONGITUD}° O")
    print(f"  Inclinación panel  : {INCLINACION_PANEL}°")
    print(f"  Declinación δ      : {res['δ [°]']}°")
    print(f"  Ángulo amanecer hs : {res['hs [°]']}°")
    print(f"  Duración del día   : {res['Duración del día [h]']} h")
    print(f"  Altitud máx. βmax  : {res['βmax [°]']}°")
    print()

    # Encabezado de tabla
    print(f"  {'Momento':<12} {'h[°]':>7} {'β[°]':>7} {'γ[°]':>8} "
          f"{'cos θ':>8} {'θ[°]':>7} {'I[W/m²]':>9} "
          f"{'Pnorm[W]':>10} {'Pabs[W]':>9}")
    print("  " + "-" * 83)

    for h_data in res["Horarios"]:
        print(
            f"  {list(HORAS_REPRESENTATIVAS.keys())[res['Horarios'].index(h_data)]:<12} "
            f"{h_data['h [°]']:>7.3f} "
            f"{h_data['β [°]']:>7.2f} "
            f"{str(h_data['γ [°]']):>8} "
            f"{h_data['cos θ']:>8.5f} "
            f"{h_data['θ [°]']:>7.2f} "
            f"{h_data['I [W/m²]']:>9.1f} "
            f"{h_data['Pnormal [W]']:>10.2f} "
            f"{h_data['Pabs [W]']:>9.2f}"
        )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  TRAYECTORIAS SOLARES — Ciudad Universitaria, CDMX")
    print("  Facultad de Ingeniería, UNAM — Energías Renovables")
    print("=" * 65)

    todas_trayectorias = []
    # Cargar TMY si está disponible
    tmy_df = load_tmy()

    for nombre, N in FECHAS.items():
        # 1. Resultados para los 3 horarios representativos
        res = calcular_para_fecha(nombre, N, tmy_df=tmy_df)
        imprimir_resultados(res)

        # 2. Trayectoria completa (12 puntos) para CSV
        tray = calcular_trayectoria_horaria(N, n_puntos=12, tmy_df=tmy_df)
        todas_trayectorias.extend(tray)

    # 3. Exportar trayectorias completas a CSV
    csv_out = os.path.join("outputs", "trayectorias_CU_CDMX.csv")
    exportar_csv(todas_trayectorias, csv_out)

    print("\nListo. Revisa el CSV para las trayectorias completas.\n")
