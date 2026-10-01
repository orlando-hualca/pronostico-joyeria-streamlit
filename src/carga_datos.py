import pandas as pd

COLUMNAS_ESPERADAS = [
    "fecha", "dia_semana", "orden", "tipo", "descripcion",
    "cantidad", "precio_unitario", "subtotal", "observacion"
]

def cargar_archivo(archivo):
    nombre = archivo.name.lower()

    if nombre.endswith(".xlsx"):
        df = pd.read_excel(archivo)
    elif nombre.endswith(".csv"):
        df = pd.read_csv(archivo)
    else:
        raise ValueError("Formato no compatible.")

    df.columns = df.columns.str.strip().str.lower()
    return df

def validar_columnas(df):
    return [c for c in COLUMNAS_ESPERADAS if c not in df.columns]