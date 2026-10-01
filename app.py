import streamlit as st

from src.carga_datos import cargar_archivo, validar_columnas
from src.preparacion import preparar_datos
from src.modelo import cargar_modelo, construir_pronostico


st.set_page_config(
    page_title="Pronóstico semanal de ventas",
    page_icon="💎",
    layout="wide"
)


@st.cache_resource
def obtener_modelo():
    return cargar_modelo()


st.title("💎 Pronóstico semanal de ventas")
st.write(
    "Estimación de unidades vendidas por categoría de producto "
    "para la siguiente semana."
)

st.divider()

archivo = st.file_uploader(
    "Cargar historial de ventas",
    type=["xlsx", "csv"]
)

if archivo is not None:
    try:
        df = cargar_archivo(archivo)
        faltantes = validar_columnas(df)

        if faltantes:
            st.error(f"El archivo no contiene las columnas necesarias: {faltantes}")
            st.stop()

        datos, datos_modelo, semanal, dataset_final = preparar_datos(df)

        preprocesador, modelo = obtener_modelo()

        resultado, semana_futura = construir_pronostico(
            semanal,
            preprocesador,
            modelo
        )

        ultima_semana = semanal["semana"].max()
        total_estimado = resultado["Unidades estimadas"].sum()

        st.success("Historial procesado correctamente.")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Última semana completa",
            ultima_semana.strftime("%d/%m/%Y")
        )

        c2.metric(
            "Semana pronosticada",
            semana_futura.strftime("%d/%m/%Y")
        )

        c3.metric(
            "Total estimado",
            f"{total_estimado} unidades"
        )

        st.divider()

        st.subheader(
            f"Pronóstico para la semana iniciada el "
            f"{semana_futura.strftime('%d/%m/%Y')}"
        )

        st.caption(
            "Las estimaciones corresponden a la semana siguiente "
            "a la última semana completa disponible en el historial."
        )

        tabla = resultado[
            ["Categoría", "Unidades estimadas"]
        ].copy()

        st.dataframe(
            tabla,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Unidades estimadas por categoría")

        grafico = tabla.set_index("Categoría")

        st.bar_chart(
            grafico,
            y="Unidades estimadas"
        )

        with st.expander("Información del historial procesado"):
            st.write(f"Registros originales: **{len(df):,}**")
            st.write(f"Registros de venta: **{len(datos):,}**")
            st.write(f"Registros clasificados: **{len(datos_modelo):,}**")
            st.write(f"Semanas completas: **{semanal['semana'].nunique()}**")
            st.write(f"Categorías: **{semanal['categoria'].nunique()}**")
            st.write(f"Observaciones para modelado: **{len(dataset_final):,}**")

    except Exception as e:
        st.error(f"No se pudo generar el pronóstico: {e}")

else:
    st.info(
        "Cargue el archivo original del historial de ventas para generar el pronóstico."
    )