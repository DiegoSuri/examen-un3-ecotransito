# ==========================================================================
# TAREA 1: INGESTA, DIAGNÓSTICO Y SUSTITUCIÓN DE CENTINELAS
# ==========================================================================
import pandas as pd
import numpy as np

# Mostrar todas las columnas en la terminal
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', None)

# 1. Cargar o definir el dataset
data = {
    'ID_Registro': ['REG-001', 'REG-002', 'REG-003', 'REG-004', 'REG-005',
                    'REG-006', 'REG-007', 'REG-008', 'REG-009', 'REG-010', 'REG-010'],
    'Timestamp': ['2026-10-01 08:00:00', '2026-10-01 09:00:00', '2026-10-01 10:00:00',
                  '2026-10-01 11:00:00', '2026-10-01 12:00:00', '2026-10-01 13:00:00',
                  '2026-10-01 14:00:00', '2026-10-01 15:00:00', '2026-10-01 16:00:00',
                  '2026-10-01 17:00:00', '2026-10-01 17:00:00'],
    'Zona_Estacion': ['Centro_Historico', 'centro_historico', 'CENTRO_HISTORICO',
                      'Zona_Oriente', 'Zona_Oriente', 'zona_oriente', 'Zona_Sur',
                      'Zona_Sur', 'Zona_Sur', 'Zona_Sur', 'Zona_Sur'],
    'Concentracion_PM25': [35.4, 42.1, -999.0, 120.5, np.nan, 310.0, 18.2, 20.1, 19.5, 19.5, 19.5],
    'Nivel_NO2': [42.1, 48.5, 51.0, 89.2, 82.0, 195.0, 22.4, -999.0, 25.1, 25.1, 25.1],
    'Indice_Trafico_Vehicular': [85.0, 92.5, 88.0, 98.0, 91.2, 15.0, 45.0, 42.1, np.nan, 41.0, 41.0],
    'Estado_Sensor': ['Normal', 'Normal', 'Alerta', 'Alerta', 'Normal', 'Falla', 'Normal',
                      'Normal', 'Normal', 'Normal', 'Normal']
}
df_raw = pd.DataFrame(data)

# ESCRIBE TU CÓDIGO AQUÍ:
# --------------------------------------------------------------------------
# a) Diagnóstico inicial de duplicados y centinelas
# --------------------------------------------------------------------------
cols_num = df_raw.select_dtypes(include=np.number).columns

print("=== DIAGNÓSTICO INICIAL ===")
print("Valores NaN por columna numérica:")
print(df_raw[cols_num].isna().sum())
print("\nValores centinela (-999.0) por columna numérica:")
print((df_raw[cols_num] == -999.0).sum())
print("\nFilas totalmente duplicadas:", df_raw.duplicated().sum())

# --------------------------------------------------------------------------
# b) Reemplazar centinela -999.0 por np.nan y eliminar duplicados
# --------------------------------------------------------------------------
df_raw = df_raw.replace(-999.0, np.nan)
df_raw = df_raw.drop_duplicates().reset_index(drop=True)

print("\n=== DESPUÉS DE LIMPIEZA INICIAL ===")
print("NaN por columna numérica (centinelas ya convertidos):")
print(df_raw[cols_num].isna().sum())
print("\nDimensiones:", df_raw.shape)
print(df_raw)


# ==========================================================================
# TAREA 2: PIPELINE DE LIMPIEZA, FEATURE ENGINEERING E IMPUTACIÓN
# ==========================================================================
from sklearn.impute import KNNImputer

df_clean = df_raw.copy()

# ESCRIBE TU CÓDIGO AQUÍ:
# --------------------------------------------------------------------------
# 1. Homogeneizar 'Zona_Estacion'
# --------------------------------------------------------------------------
df_clean['Zona_Estacion'] = df_clean['Zona_Estacion'].str.strip().str.title()

# --------------------------------------------------------------------------
# 2. Convertir Timestamp a datetime y extraer Hora_Dia
# --------------------------------------------------------------------------
df_clean['Timestamp'] = pd.to_datetime(df_clean['Timestamp'])
df_clean['Hora_Dia'] = df_clean['Timestamp'].dt.hour

# --------------------------------------------------------------------------
# 3. Detección de outliers por IQR en 'Concentracion_PM25' y sustitución por np.nan
# --------------------------------------------------------------------------
col = 'Concentracion_PM25'
Q1 = df_clean[col].quantile(0.25)
Q3 = df_clean[col].quantile(0.75)
IQR = Q3 - Q1
limite_inf = Q1 - 1.5 * IQR
limite_sup = Q3 + 1.5 * IQR

mascara_outliers = (df_clean[col] < limite_inf) | (df_clean[col] > limite_sup)
print(f"\nQ1={Q1:.2f}  Q3={Q3:.2f}  IQR={IQR:.2f}")
print(f"Límites IQR: [{limite_inf:.2f}, {limite_sup:.2f}]")
print("Outliers detectados:")
print(df_clean.loc[mascara_outliers, ['ID_Registro', col]])

df_clean.loc[mascara_outliers, col] = np.nan

# --------------------------------------------------------------------------
# 4. Imputación avanzada k-NN sobre variables numéricas
# --------------------------------------------------------------------------
cols_imputar = ['Concentracion_PM25', 'Nivel_NO2', 'Indice_Trafico_Vehicular']
imputer = KNNImputer(n_neighbors=2)
df_clean[cols_imputar] = imputer.fit_transform(df_clean[cols_imputar])
df_clean[cols_imputar] = df_clean[cols_imputar].round(2)

print("\n=== df_clean FINAL ===")
print(df_clean)
print("\nNaN restantes:", df_clean.isna().sum().sum())

# Guardar para usarlo en la app de Streamlit (Tarea 4)
df_clean.to_csv("cdmx_air_traffic_clean.csv", index=False)


# ==========================================================================
# TAREA 3: EDA Y ANÁLISIS DE CORRELACIÓN
# ==========================================================================
# ESCRIBE TU CÓDIGO AQUÍ:
# --------------------------------------------------------------------------
# 1. Estadísticas de forma para Concentracion_PM25
# --------------------------------------------------------------------------
pm = df_clean['Concentracion_PM25']
print("\n=== ESTADÍSTICAS DE Concentracion_PM25 ===")
print(f"Media:               {pm.mean():.2f}")
print(f"Mediana:             {pm.median():.2f}")
print(f"Desviación estándar: {pm.std():.2f}")
print(f"Skewness:            {pm.skew():.2f}")
print(f"Kurtosis:            {pm.kurt():.2f}")

# --------------------------------------------------------------------------
# 2. Matrices de correlación Pearson y Spearman
# --------------------------------------------------------------------------
num_cols = ['Concentracion_PM25', 'Nivel_NO2', 'Indice_Trafico_Vehicular']
corr_pearson = df_clean[num_cols].corr(method='pearson')
corr_spearman = df_clean[num_cols].corr(method='spearman')

print("\n=== CORRELACIÓN DE PEARSON ===")
print(corr_pearson.round(3))
print("\n=== CORRELACIÓN DE SPEARMAN ===")
print(corr_spearman.round(3))

# Pares con |r| >= 0.85
print("\n=== PARES CON |r| >= 0.85 ===")
for nombre, matriz in [("Pearson", corr_pearson), ("Spearman", corr_spearman)]:
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            r = matriz.iloc[i, j]
            if abs(r) >= 0.85:
                print(f"{nombre}: {num_cols[i]} vs {num_cols[j]} -> r = {r:.3f}")
