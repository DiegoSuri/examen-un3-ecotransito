# ==========================================================================
# TAREA 4: SCRIPT DE STREAMLIT (app_ecotransito.py)
# ==========================================================================
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Configuración de página
st.set_page_config(page_title="Sistema EcoTránsito CDMX", layout="wide")
st.title("🌎 Sistema EcoTránsito CDMX")
st.caption("Diagnóstico de calidad del aire y flujo vehicular")


@st.cache_data
def cargar_datos():
    return pd.read_csv("cdmx_air_traffic_clean.csv", parse_dates=["Timestamp"])


df = cargar_datos()

# 2. Carga y filtrado de datos con barra lateral
st.sidebar.header("Filtros")
zonas = sorted(df["Zona_Estacion"].unique())
zonas_sel = st.sidebar.multiselect("Zonas de monitoreo", options=zonas, default=zonas)

df_f = df[df["Zona_Estacion"].isin(zonas_sel)]

if df_f.empty:
    st.warning("Selecciona al menos una zona en la barra lateral para ver los datos.")
    st.stop()

# 3. Métricas principales (KPIs)
c1, c2, c3 = st.columns(3)
c1.metric("Total de lecturas activas", len(df_f))
c2.metric("Promedio PM2.5", f"{df_f['Concentracion_PM25'].mean():.1f} μg/m³")
c3.metric("Índice máximo de tráfico", f"{df_f['Indice_Trafico_Vehicular'].max():.1f}")

# 4. Pestañas interactivas de visualización (Plotly + Seaborn)
tab1, tab2 = st.tabs(["📈 Dispersión Tráfico vs PM2.5", "🔥 Mapa de calor de correlación"])

with tab1:
    fig = px.scatter(
        df_f,
        x="Indice_Trafico_Vehicular",
        y="Concentracion_PM25",
        color="Zona_Estacion",
        hover_data=["ID_Registro", "Nivel_NO2", "Estado_Sensor"],
        title="Relación entre tráfico vehicular y PM2.5",
        labels={
            "Indice_Trafico_Vehicular": "Índice de tráfico vehicular",
            "Concentracion_PM25": "PM2.5 (μg/m³)",
            "Zona_Estacion": "Zona",
        },
    )
    st.plotly_chart(fig)

with tab2:
    cols_num = ["Concentracion_PM25", "Nivel_NO2", "Indice_Trafico_Vehicular"]
    if len(df_f) < 3:
        st.info("Se necesitan al menos 3 lecturas para calcular la correlación.")
    else:
        corr = df_f[cols_num].corr()
        fig2, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f", ax=ax)
        ax.set_title("Matriz de correlación (Pearson)")
        st.pyplot(fig2)
