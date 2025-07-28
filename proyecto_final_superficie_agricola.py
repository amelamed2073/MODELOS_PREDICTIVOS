# UNIVERSIDAD TECNOLÓGICA DE PANAMÁ
# FACULTAD DE INGENIERÍA INDUSTRIAL
# Modelado Predictivo del Uso Agrícola de la Tierra en Panamá
# AARON MELAMED
# Fecha: 2024-07-28
# PROFESOR: Juan Marcos Castilo, Phd
 
"""
 DESCRIPCIÓN DE LOS DATASETS UTILIZADOS:

1. datos_principales (Insumos_TierraUso_S_Todos_los_Datos.csv):
   - Contiene los valores anuales del uso de la tierra por país, producto (tipo de cobertura), elemento (tipo de medición) y unidad.
   - Incluye valores desde 1961 hasta 2023 en columnas anuales (Y1961, Y1962, ..., Y2023).
   - incluye columnas auxiliares por año que indican el tipo de fuente o calidad del dato.

2. cod_areas (Insumos_TierraUso_S_C¢digodel†reas.csv):
   - Tabla de referencia que asocia el código de área de la FAO con el nombre del país o región.
   - Permite mapear cada país a su continente o agregado regional.

3. elementos (Insumos_TierraUso_S_Elementos.csv):
   - Catálogo que describe los tipos de elementos medidos (por ejemplo: superficie, porcentaje, carbono, etc.).
   - Es útil para interpretar los nombres técnicos en la columna "Elemento".

4. productos (Insumos_TierraUso_S_C¢digodelproductos.csv):
   - Tabla dela correspondencia entre códigos de productos y su descripción.

5. simbolos (Insumos_TierraUso_S_S°mbolos.csv):
   - Diccionario de banderas de calidad y fuente de los datos (A = oficial, E = estimado, I = imputado, F = Pronóstico, N = Datos no Disponibles.).
   - Permite hacer filtrado o evaluación de confiabilidad de los datos en el análisis.
"""

import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os
import warnings
warnings.filterwarnings("ignore", message="No frequency information was provided")
os.makedirs("modelos", exist_ok=True)

# Archivos de entrada
archivos = {
    "datos_principales": "Insumos_TierraUso_S_Todos_los_Datos.csv",
    "cod_areas": "Insumos_TierraUso_S_C¢digodel†reas.csv",
    "elementos": "Insumos_TierraUso_S_Elementos.csv",
    "productos": "Insumos_TierraUso_S_C¢digodelproductos.csv",
    "simbolos": "Insumos_TierraUso_S_S°mbolos.csv"
}


datasets = {nombre: pd.read_csv(ruta, encoding='utf-8').rename(columns=lambda x: x.strip()) for nombre, ruta in archivos.items()}
cod_areas = datasets["cod_areas"]

# Normalización de acentos y limpieza para los nombres de países
import unicodedata

def normalizar_texto(texto):
    if isinstance(texto, str):
        texto = unicodedata.normalize('NFKD', texto) # Normalizar acentos
        texto = ''.join([c for c in texto if not unicodedata.combining(c)]) # Eliminar acentos
        return texto
    return texto


# Normalizar nombres de columnas eliminando espacios y acentos
cod_areas.columns = cod_areas.columns.str.strip() # Eliminar espacios al inicio y final
cod_areas.columns = cod_areas.columns.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8') # Eliminar acentos
print("Columnas cod_areas tras normalización:", cod_areas.columns.tolist())
cod_areas['Area'] = cod_areas['Area'].apply(normalizar_texto) # Normalizar nombres de países en cod_areas

# Mostrar estructura de cada dataset con describe
print("\nDESCRIPCIÓN ESTADÍSTICA DE CADA DATASET:")
for nombre, df_temp in datasets.items(): # Normalizar nombres de columnas
    print(f"\n--- {nombre.upper()} ---")
    print(df_temp.describe(include='all'))

# Mostrar número de columnas por dataset
print("\nCantidad de columnas por dataset:")
for nombre, df_temp in datasets.items():
    print(f"{nombre}: {df_temp.shape[1]} columnas")



# Acceso directo a cada dataframe
df = datasets["datos_principales"] # Datos principales
elementos = datasets["elementos"] # catálogo de elementos
simbolos = datasets["simbolos"] # diccionario de símbolos

# Filtrar columnas útiles (excluir banderas y notas)
columnas_utiles = [col for col in df.columns if not col.endswith('F') and not col.endswith('N')] # Excluir columnas que terminan en 'F' o 'N'
df = df[columnas_utiles] # Normalizar nombres de columnas

# Mostrar columnas cod_areas para depurar
print("\nColumnas disponibles en cod_areas (tras limpieza):")
print(cod_areas.columns.tolist())

cod_areas.columns = cod_areas.columns.str.strip() # Eliminar espacios al inicio y final
if 'Area' not in cod_areas.columns: # Si 'Area' no está en cod_areas, lanzar error
    raise KeyError(f"'Area' no encontrada en cod_areas. Columnas disponibles: {cod_areas.columns.tolist()}")
assert 'Area' in cod_areas.columns, f"Columna 'Area' no encontra da. Columnas disponibles: {cod_areas.columns.tolist()}"

# Unir con códigos de áreas
cod_areas.rename(columns=lambda x: x.strip(), inplace=True) # Eliminar espacios en nombres de columnas
df_merged = df.merge( # Unir df con cod_areas
    cod_areas[['Codigo del area', 'Area']],
    left_on='Código del área',
    right_on='Codigo del area',
    how='left',
    suffixes=('', '_area')
)

# Validar existencia de la columna 'Area' (puede ser 'Area' o 'Area_y' tras el merge)
area_col = 'Area' if 'Area' in df_merged.columns else ('Area_y' if 'Area_y' in df_merged.columns else None) # Determinar el nombre correcto de la columna 'Area'
if not area_col:
    raise KeyError(f"'Area' no encontrada en df_merged. Columnas disponibles: {df_merged.columns.tolist()}") # Asegurar que 'Area' existe en df_merged



# Diccionario de correcciones manuales de nombres de países problemáticos
correcciones_paises = {
    "Côte d'Ivoire": "Ivory Coast",
    "Cabo Verde": "Cape Verde",
    "Republica Centroafricana": "Central African Republic",
    "Sudán del Sur": "South Sudan",
    "Palestina": "Palestine, State of",
    "Territorio Palestino Ocupado": "Palestine, State of",
    "Federación de Rusia": "Russia",
    "Republica de Moldova": "Moldova",
    "Viet Nam": "Vietnam",
    "República de Corea": "South Korea",
    "Corea, República Popular Democrática de": "North Korea",
    "Estados Unidos de América": "United States",
    "Reino Unido de Gran Bretaña e Irlanda del Norte": "United Kingdom",
    "República Árabe Siria": "Syria",
    "Irán (República Islámica del)": "Iran",
    "Venezuela (República Bolivariana de)": "Venezuela",
    "Bolivia (Estado Plurinacional de)": "Bolivia",
    "Tanzania, República Unida de": "Tanzania",
    "Mundo": None,
    "Otros países de África": None,
    "Otros países de Asia": None,
    "Otros países de Europa": None,
    "Otros países de América": None,
    "Otros países de Oceanía": None
}

df_merged['País'] = df_merged[area_col].apply(lambda x: normalizar_texto(str(x).strip()))

# Filtrar Tierras agrícolas, Superficie, 1000 ha
df_filtrado = df_merged[
    (df_merged["Producto"] == "Tierras agrícolas") &
    (df_merged["Elemento"] == "Superficie") &
    (df_merged["Unidad"] == "1000 ha")
]

df_limpio = df_filtrado.copy()

print(df_limpio.head())
print(df_limpio.describe())

print("\nMuestra de registros del dataframe df_limpio:")
print(df_limpio.head(10))

df_limpio.to_csv("Superficie_Agricola_Limpia.csv", index=False)
print("\nArchivo 'Superficie_Agricola_Limpia.csv' guardado exitosamente con todas las columnas anuales.")

# --- Gráficas de evolución mundial y Panamá ---
import matplotlib.pyplot as plt

# Asegurar que los años estén correctamente seleccionados
columnas_anuales = [col for col in df_limpio.columns if col.startswith("Y") and len(col) == 5]

# Gráfico 1: Evolución mundial
df_mundial = df_limpio[columnas_anuales].sum().sort_index() # datos mundiales sumando todas las superficies agrícolas --- fuente df_limpio filtro por columnas anuales
plt.figure(figsize=(12, 6))
df_mundial.plot(marker='o')
plt.title("#1 Superficie agrícola total en el mundo (1000 ha)")
plt.xlabel("Año")
plt.ylabel("Superficie agrícola (1000 ha)")
plt.grid(True)
plt.tight_layout()
plt.savefig("#1 grafico_mundial_superficie_agricola.png")
plt.close()

# Gráfico 2: Evolución Panamá
df_panama = df_limpio[df_limpio["País"] == "Panama"] # datos de Panamá --- fuente df_limpio filtro Panama
if not df_panama.empty:
    superficie_panama = df_panama[columnas_anuales].sum().sort_index()
    plt.figure(figsize=(12, 6))
    superficie_panama.plot(marker='o', color='green')
    plt.title("#2 Superficie agrícola en Panamá (1000 ha)")
    plt.xlabel("Año")
    plt.ylabel("Superficie agrícola (1000 ha)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("#2 grafico_panama_superficie_agricola.png")
    plt.close()

    # Listado de superficie agrícola en Panamá por año
    print("\nSuperficie agrícola en Panamá por año (1000 ha):")
    print(superficie_panama)

    superficie_panama.to_csv("superficie_agricola_panama_por_anio.csv", header=['Superficie (1000 ha)'])

    # Gráfico 4: Promedio de superficie agrícola en Panamá (últimos 5 años)
    promedio_panama = df_panama[columnas_anuales[-5:]].mean().sort_index()

    plt.figure(figsize=(12, 6))
    promedio_panama.plot(marker='o', color='orange')
    plt.title("#3 Promedio superficie agrícola en Panamá (últimos 5 años)")
    plt.xlabel("Año")
    plt.ylabel("Superficie agrícola (1000 ha)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("#3 grafico_promedio_panama_superficie_agricola.png")
    plt.close()
else:
    print("No se encontraron registros para Panamá en df_limpio.")


# ===============================
# PRONÓSTICO DE SERIES DE TIEMPO
# ===============================

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluar_modelo(y_real, y_pred):
    y_real = np.array(y_real)
    y_pred = np.array(y_pred)

    if len(y_real) != len(y_pred):
        raise ValueError(f"Las longitudes de y_real ({len(y_real)}) y y_pred ({len(y_pred)}) no coinciden.")

    mae = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    mape = np.mean(np.abs((y_real - y_pred) / y_real)) * 100
    # r2 = r2_score(y_real, y_pred)

    # if r2 > 1 or r2 < -1:
    #     print(f"⚠️ Advertencia: valor R² inesperado ({r2:.4f}). Verifica los datos de entrada.")

    # return {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape, "R²": r2}
    return {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape}

# Preparar series temporales
serie_panama = df_limpio[df_limpio["País"] == "Panama"][columnas_anuales].sum() # datos de Panamá --- fuente df_limpio filtro Panama
serie_panama.index = serie_panama.index.str[1:].astype(int) # Convertir índices de columnas anuales a enteros
serie_panama = serie_panama.sort_index() # Asegurar que los índices estén ordenados

serie_mundial = df_limpio[columnas_anuales].sum() # datos mundiales sumando todas las superficies agrícolas --- fuente df_limpio filtro por columnas anuales
serie_mundial.index = serie_mundial.index.str[1:].astype(int) # Convertir índices de columnas anuales a enteros
serie_mundial = serie_mundial.sort_index() # Asegurar que los índices estén ordenados

# ===== Pronóstico usando Holt-Winters ====
def forecast_holt(series, n_years=5, nombre=""): 
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.date_range(start=f"{series.index[0]}-12-31", periods=len(series), freq='YE')
    modelo = ExponentialSmoothing(series, trend="add", seasonal=None).fit()
    joblib.dump(modelo, f"modelos/modelo_holt_{nombre.lower()}.pkl")
    pred = modelo.forecast(n_years)
    pred.index = pd.to_datetime([f"{a}-12-31" for a in range(2024, 2024 + n_years)])
    return pred

# ===== Pronóstico usando ARIMA ====
def forecast_arima(series, n_years=5, nombre=""):
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.date_range(start=f"{series.index[0]}-12-31", periods=len(series), freq='YE')
    modelo = ARIMA(series, order=(1,1,1)).fit()
    joblib.dump(modelo, f"modelos/modelo_arima_{nombre.lower()}.pkl")
    pred = modelo.forecast(n_years)
    pred.index = pd.to_datetime([f"{a}-12-31" for a in range(2024, 2024 + n_years)])
    return pred

# ===== Pronóstico usando Prophet ====
def forecast_prophet(series, n_years=5, nombre=""):
    df = series.reset_index()
    df.columns = ['ds', 'y']
    df['ds'] = pd.to_datetime(df['ds'], format='%Y')
    import warnings
    warnings.filterwarnings("ignore", message="'Y' is deprecated and will be removed in a future version")
    modelo = Prophet(yearly_seasonality=False)
    modelo.fit(df)
    joblib.dump(modelo, f"modelos/modelo_prophet_{nombre.lower()}.pkl")
    futuro = modelo.make_future_dataframe(periods=n_years, freq='Y', include_history=False)
    pronostico = modelo.predict(futuro)
    return pronostico.set_index('ds')['yhat'].iloc[-n_years:]

# ===== Pronóstico usando Random Forest con ajuste de hiperparámetros ====
def forecast_random_forest(series, n_years=5, nombre=""):
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.date_range(start=f"{series.index[0]}-12-31", periods=len(series), freq='YE')
    # Preparar DataFrame para GridSearch
    df_rf = pd.DataFrame({'value': series.values}, index=series.index)
    df_rf['year_num'] = df_rf.index.year
    X = df_rf[['year_num']]
    y = df_rf['value']
    # Separar entrenamiento y validación
    X_train = X.loc[:'2012']
    y_train = y.loc[:'2012']
    X_valid = X.loc['2013':'2023']
    y_valid = y.loc['2013':'2023']
    # Hiperparámetros y validación temporal
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [2, 5, 10, None],
        'min_samples_split': [2, 5, 10]
    }
    tscv = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=tscv, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_
    joblib.dump(model, f"modelos/modelo_rf_{nombre.lower()}.pkl")
    # Predicción y métricas
    y_pred = model.predict(X_valid)
    print("Mejores Hiperparámetros:", grid_search.best_params_)
    print("R²:", r2_score(y_valid, y_pred))
    print("MAE:", mean_absolute_error(y_valid, y_pred))
    import numpy as np  # Asegúrate de tener esta importación al inicio del archivo si no está ya
    print("RMSE:", np.sqrt(mean_squared_error(y_valid, y_pred)))
    # Pronóstico futuro
    future_years = [series.index[-1].year + i + 1 for i in range(n_years)]
    X_future = pd.DataFrame({'year_num': future_years})
    pred = model.predict(X_future)
    fechas_futuras = pd.to_datetime([f"{a}-12-31" for a in future_years])
    return pd.Series(pred, index=fechas_futuras)


# ===============================
# aplicar modelos a las series temporales de Panamá y mundial
# ===============================
import matplotlib.pyplot as plt
def aplicar_modelos(nombre, series):    
    print(f"\n--- PRONÓSTICOS PARA {nombre.upper()} ---")
    # Separar en entrenamiento y prueba
    serie_entrenamiento = series.iloc[:-5].copy() # Usar todos los años excepto los últimos 5 para entrenamiento
    serie_validacion = series.iloc[-5:].copy() # Usar los últimos 5 años para validación
    # Asegurar que los índices sean DatetimeIndex
    if not isinstance(serie_entrenamiento.index, pd.DatetimeIndex):
        serie_entrenamiento.index = pd.date_range(start=f"{serie_entrenamiento.index[0]}-12-31", periods=len(serie_entrenamiento), freq='YE')
    if not isinstance(serie_validacion.index, pd.DatetimeIndex):
        serie_validacion.index = pd.date_range(start=f"{serie_validacion.index[0]}-12-31", periods=len(serie_validacion), freq='YE')

    n_validacion = len(serie_validacion)
    forecast_hw = forecast_holt(serie_entrenamiento, n_validacion, nombre)
    forecast_ar = forecast_arima(serie_entrenamiento, n_validacion, nombre)
    forecast_pr = forecast_prophet(serie_entrenamiento, n_validacion, nombre)
    forecast_rf = forecast_random_forest(serie_entrenamiento, n_validacion, nombre)

# Evaluar los modelos con la serie de validación utilizando las métricas definidas 2021 a 2023
    evaluaciones = {
        "Holt-Winters": evaluar_modelo(serie_validacion.values, forecast_hw.values),
        "ARIMA": evaluar_modelo(serie_validacion.values, forecast_ar.values),
        "Prophet": evaluar_modelo(serie_validacion.values, forecast_pr.values),
        "Random Forest": evaluar_modelo(serie_validacion.values, forecast_rf.values)
    }

    print(f"\n--- MÉTRICAS DE EVALUACIÓN (2021–2023) PARA {nombre.upper()} ---")
    for modelo, metricas in evaluaciones.items():
        print(f"\n{modelo}")
        for metrica, valor in metricas.items():
            if metrica != "R²":
                print(f"{metrica}: {valor:.2f}")

    # --- EVALUACIÓN ROBUSTA (Z-SCORE Y BENCHMARK CON MEDIA) ---
    from sklearn.metrics import r2_score
    from scipy.stats import zscore

    def evaluar_modelo_extendido(y_true, y_pred, nombre_modelo):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred) ** 0.5
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        # R² clásico
        r2 = r2_score(y_true, y_pred)

        # Verificación para evitar zscore con desviación estándar cero
        if np.std(y_true) == 0 or np.std(y_pred) == 0:
            r2_z = np.nan
            # print(f"⚠️ Advertencia: R² Z-Score no se puede calcular para {nombre_modelo} (desviación estándar = 0)")
        else:
            try:
                r2_z = r2_score(zscore(y_true), zscore(y_pred))
            except Exception as e:
                r2_z = np.nan
                # print(f"⚠️ Error al calcular R² Z-Score para {nombre_modelo}: {e}")

        # Benchmark: modelo de promedio histórico
        baseline = np.full_like(y_true, np.mean(y_true))
        r2_baseline = r2_score(y_true, baseline)

        print(f"\n--- MÉTRICAS EXTENDIDAS PARA {nombre_modelo} ---")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAPE (%): {mape:.2f}")
        # print(f"R²: {r2:.4f}")
        # print(f"R² Z-Score (forma): {r2_z:.4f}" if not np.isnan(r2_z) else "R² Z-Score (forma): No disponible")
        # print(f"R² vs. Media histórica: {r2_baseline:.4f} (mejora sobre baseline)")
        return r2

    # Aplica la evaluación extendida a cada modelo en la validación de Panamá
    print("\n--- VALIDACIÓN ROBUSTA (PANAMÁ) 2021–2023 ---")
    _ = evaluar_modelo_extendido(serie_validacion.values, forecast_hw.values, "Holt-Winters")
    _ = evaluar_modelo_extendido(serie_validacion.values, forecast_ar.values, "ARIMA")
    _ = evaluar_modelo_extendido(serie_validacion.values, forecast_pr.values, "Prophet")
    _ = evaluar_modelo_extendido(serie_validacion.values, forecast_rf.values, "Random Forest")

    # Gráfico comparativo de métricas (MAE, RMSE, MAPE, R2)
    import seaborn as sns

    df_eval = pd.DataFrame(evaluaciones).T
    df_eval.reset_index(inplace=True)
    df_eval.rename(columns={'index': 'Modelo'}, inplace=True)
    
    plt.figure(figsize=(12, 6))
    df_eval_melted = df_eval.melt(id_vars="Modelo", var_name="Métrica", value_name="Valor")
    sns.barplot(data=df_eval_melted, x="Métrica", y="Valor", hue="Modelo")
    plt.title(f"Evaluación de modelos para {nombre.upper()} (2021–2023)")
    plt.ylabel("Valor")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"evaluacion_modelos_{nombre.lower()}.png")
    plt.close()

    # Pronóstico final con serie completa para 2024-2026
    forecast_hw_final = forecast_holt(series, n_years=3, nombre=nombre)
    forecast_ar_final = forecast_arima(series, n_years=3, nombre=nombre)
    forecast_pr_final = forecast_prophet(series, n_years=3, nombre=nombre)
    forecast_rf_final = forecast_random_forest(series, n_years=3, nombre=nombre)

    df_resultado = pd.DataFrame({
        "Holt-Winters": forecast_hw_final,
        "ARIMA": forecast_ar_final,
        "Prophet": forecast_pr_final,
        "Random Forest": forecast_rf_final
    })
    print(f"\n--- PRONÓSTICOS PARA {nombre.upper()} (2024–2026) ---")
    print(df_resultado)
    df_resultado.to_csv(f"pronostico_{nombre.lower()}.csv")
    df_resultado.plot(title=f"Pronóstico 2024–2026: {nombre}", marker='o')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"pronostico_{nombre.lower()}.png")
    plt.close()


aplicar_modelos("Panama", serie_panama)
aplicar_modelos("Mundial", serie_mundial)


# ===============================
# PRONÓSTICO POR PRODUCTO EN PANAMÁ (2024–2028) CON TODOS LOS MODELOS
# ===============================
print("\n=== PRONÓSTICO POR PRODUCTO EN PANAMÁ (2024–2028) CON TODOS LOS MODELOS ===")

# === ENTRENAMIENTO POR PRODUCTO EN PANAMÁ ===
df_panama = df_limpio[df_limpio["Area"] == "Panama"]
productos_relevantes = df_panama["Producto"].value_counts().nlargest(10).index.tolist()

datos_productos = {}
for producto in productos_relevantes:
    serie = df_panama[df_panama["Producto"] == producto]
    serie = serie[[col for col in serie.columns if col.startswith("Y")]].T
    serie.columns = [producto]
    serie.index = pd.to_datetime([col[1:] + "-12-31" for col in serie.index])
    datos_productos[producto] = serie.dropna()

# Para mantener compatibilidad con la estructura previa  usamos datos_productos 

for producto, df_serie in datos_productos.items():
    # df_serie es un DataFrame de una columna, con índice datetime
    serie = df_serie[producto]
    serie = serie.sort_index()

    pred_holt = forecast_holt(serie, n_years=5, nombre=producto)
    pred_arima = forecast_arima(serie, n_years=5, nombre=producto)
    pred_prophet = forecast_prophet(serie, n_years=5, nombre=producto)
    pred_rf = forecast_random_forest(serie, n_years=5, nombre=producto)

    df_modelos = pd.DataFrame({
        "Holt-Winters": pred_holt,
        "ARIMA": pred_arima,
        "Prophet": pred_prophet,
        "Random Forest": pred_rf
    })

    # Index de años (2024–2028)
    df_modelos.index = [a.year for a in df_modelos.index]
    # Exportar a CSV individual
    nombre_archivo = f"pronostico_5_anos_{producto.lower().replace(' ', '_')}"
    df_modelos.to_csv(f"{nombre_archivo}.csv")

    # Graficar todas las predicciones
    plt.figure(figsize=(12, 6))
    for modelo in df_modelos.columns:
        plt.plot(df_modelos.index, df_modelos[modelo], marker='o', label=modelo)
    plt.title(f"Pronóstico (2024–2028) para {producto} - Todos los Modelos")
    plt.xlabel("Año")
    plt.ylabel("Producción estimada")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{nombre_archivo}.png")
    plt.close()

# Consolidar pronósticos de los 10 productos principales en una sola gráfica
print("\nGenerando gráfica consolidada para los 10 productos principales...")
plt.figure(figsize=(14, 8))

for producto in datos_productos.keys():
    nombre_archivo = f"pronostico_5_anos_{producto.lower().replace(' ', '_')}.csv"
    try:
        df_producto = pd.read_csv(nombre_archivo, index_col=0)
        # Seleccionar el modelo Holt-Winters para consistencia (puedes cambiarlo)
        if "Holt-Winters" in df_producto.columns:
            plt.plot(df_producto.index, df_producto["Holt-Winters"], marker='o', label=producto)
    except Exception as e:
        print(f"Error al procesar {producto}: {e}")

plt.title("Pronóstico 2024–2028 de los 10 productos más relevantes en Panamá (Modelo Holt-Winters)")
plt.xlabel("Año")
plt.ylabel("Producción estimada")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("pronostico_consolidado_10_productos_panama.png")
plt.close()