import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

CATEGORIAS = [
    "Anillos", "Aretes", "Cadenas", "Cajas/Empaques",
    "Charms", "Collares", "Dijes",
    "Juegos/Conjuntos", "Manillas"
]

VARIABLES = [
    "mes", "semana_anio",
    "lag_1", "lag_2", "lag_4",
    "media_4", "media_8", "categoria"
]


def cargar_modelo():
    paquete = joblib.load("models/modelo_xgboost_semanal.pkl")

    preprocesador = paquete["preprocesador"]

    modelo = xgb.Booster()
    modelo.load_model(bytearray(paquete["booster_raw"]))

    return preprocesador, modelo


def construir_pronostico(semanal, preprocesador, modelo):
    ultima_semana = semanal["semana"].max()
    semana_futura = ultima_semana + pd.Timedelta(days=7)

    filas = []

    for categoria in CATEGORIAS:
        historial = (
            semanal[semanal["categoria"] == categoria]
            .sort_values("semana")["unidades"]
            .reset_index(drop=True)
        )

        fila = {
            "mes": semana_futura.month,
            "semana_anio": int(semana_futura.isocalendar().week),
            "lag_1": historial.iloc[-1],
            "lag_2": historial.iloc[-2],
            "lag_4": historial.iloc[-4],
            "media_4": historial.iloc[-4:].mean(),
            "media_8": historial.iloc[-8:].mean(),
            "categoria": categoria
        }

        filas.append(fila)

    X_futuro = pd.DataFrame(filas)

    X_procesado = preprocesador.transform(
        X_futuro[VARIABLES]
    )

    predicciones = modelo.predict(
        xgb.DMatrix(X_procesado)
    )

    predicciones = np.maximum(predicciones, 0)

    resultado = pd.DataFrame({
        "Categoría": X_futuro["categoria"],
        "Pronóstico": predicciones.round(2),
        "Unidades estimadas": predicciones.round().astype(int)
    })

    return resultado, semana_futura