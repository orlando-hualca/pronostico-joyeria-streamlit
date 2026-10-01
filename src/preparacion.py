import pandas as pd
import unicodedata


def normalizar_texto(texto):
    texto = str(texto).lower().strip()
    texto = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in texto if not unicodedata.combining(c))


def clasificar_producto(texto):
    if "juego" in texto or "conjunto" in texto:
        return "Juegos/Conjuntos"
    if "cadena" in texto and "dije" in texto:
        return "Collares"
    if "cadena" in texto:
        return "Cadenas"
    if "anillo" in texto or "argolla" in texto or "aro" in texto:
        return "Anillos"
    if any(x in texto for x in ["arete", "areta", "areto", "aretito"]):
        return "Aretes"
    if any(x in texto for x in ["manilla", "pulsera", "denario", "dinario", "esclava"]):
        return "Manillas"
    if "charm" in texto:
        return "Charms"
    if "dije" in texto or "inicial" in texto:
        return "Dijes"
    if "collar" in texto or "rosario" in texto:
        return "Collares"
    if "caja" in texto or "cajita" in texto or "bolsa" in texto:
        return "Cajas/Empaques"
    return "Otros"


def preparar_datos(df):
    datos = df.copy()

    datos["fecha"] = pd.to_datetime(
        datos["fecha"],
        format="%d/%m/%Y",
        errors="coerce"
    )

    datos = datos[
        (datos["tipo"] == "venta")
        & datos["cantidad"].notna()
        & (datos["cantidad"] > 0)
    ].copy()

    datos = datos[["fecha", "descripcion", "cantidad"]]

    datos["descripcion_norm"] = datos["descripcion"].apply(normalizar_texto)
    datos["categoria"] = datos["descripcion_norm"].apply(clasificar_producto)

    datos_modelo = datos[
        datos["categoria"] != "Otros"
    ].copy()

    datos_modelo["semana"] = (
        datos_modelo["fecha"]
        - pd.to_timedelta(datos_modelo["fecha"].dt.dayofweek, unit="D")
    )

    fecha_min = datos_modelo["fecha"].min()
    fecha_max = datos_modelo["fecha"].max()

    datos_modelo = datos_modelo[
        (datos_modelo["semana"] >= fecha_min)
        & (datos_modelo["semana"] + pd.Timedelta(days=6) <= fecha_max)
    ].copy()

    semanal = (
        datos_modelo
        .groupby(["semana", "categoria"])["cantidad"]
        .sum()
        .reset_index(name="unidades")
    )

    semanas = semanal["semana"].sort_values().unique()
    categorias = semanal["categoria"].sort_values().unique()

    indice = pd.MultiIndex.from_product(
        [semanas, categorias],
        names=["semana", "categoria"]
    )

    semanal = (
        semanal
        .set_index(["semana", "categoria"])
        .reindex(indice, fill_value=0)
        .reset_index()
    )

    semanal["mes"] = semanal["semana"].dt.month
    semanal["semana_anio"] = semanal["semana"].dt.isocalendar().week.astype(int)

    semanal = semanal.sort_values(["categoria", "semana"]).copy()

    semanal["lag_1"] = semanal.groupby("categoria")["unidades"].shift(1)
    semanal["lag_2"] = semanal.groupby("categoria")["unidades"].shift(2)
    semanal["lag_4"] = semanal.groupby("categoria")["unidades"].shift(4)

    semanal["media_4"] = (
        semanal.groupby("categoria")["unidades"]
        .transform(lambda x: x.shift(1).rolling(4).mean())
    )

    semanal["media_8"] = (
        semanal.groupby("categoria")["unidades"]
        .transform(lambda x: x.shift(1).rolling(8).mean())
    )

    variables_hist = [
        "lag_1", "lag_2", "lag_4",
        "media_4", "media_8"
    ]

    dataset_final = (
        semanal
        .dropna(subset=variables_hist)
        .sort_values(["semana", "categoria"])
        .reset_index(drop=True)
    )

    return datos, datos_modelo, semanal, dataset_final