# UTP MODELOS_PREDICTIVOS

# Modelado Predictivo del Uso Agrícola de la Tierra en Panamá

##  Descripción del Proyecto

Este proyecto tiene como objetivo desarrollar un modelo predictivo que permita proyectar la evolución de la superficie agrícola en Panamá hasta el año 2030. Se utilizaron datos históricos de FAOSTAT (1961–2023) para entrenar y evaluar diversos modelos de series temporales y algoritmos de machine learning. El resultado busca aportar insumos para políticas públicas, planificación agropecuaria y análisis territorial.

##  Motivación

Panamá enfrenta transformaciones aceleradas del uso de su territorio, principalmente por expansión urbana, degradación ambiental y cambios en el modelo de producción agropecuaria. La capacidad de anticipar cambios en la superficie agrícola es clave para garantizar seguridad alimentaria, sostenibilidad ecológica y eficiencia en el uso del suelo.

##  Dataset

- Fuente: [FAOSTAT](https://www.fao.org/faostat)
- Archivo principal: `Insumos_TierraUso_S_Todos_los_Datos.csv`
- Atributos claves:
  - `Area` (país/región)
  - `Producto` (ej. tierras agrícolas)
  - `Elemento` (ej. superficie)
  - `Unidad` (1000 ha)
  - Años: `Y1961` a `Y2023`
- Limpieza y transformación:
  - Filtros por producto, elemento y unidad
  - Conversión a formato largo (tidy)
  - Unificación de países, regiones y símbolos

##  Análisis Exploratorio

Se realizaron visualizaciones con `matplotlib` para:
- Tendencia histórica global y nacional
- Comparación por década
- Variaciones recientes y detección de outliers

Gráficos generados:
- `grafico_mundial_superficie_agricola.png`
- `grafico_panama_superficie_agricola.png`
- `grafico_promedio_panama_superficie_agricola.png`

##  Modelos Utilizados

Se aplicaron y compararon cuatro enfoques de modelado:

1. **Holt-Winters (suavizado exponencial)**
2. **ARIMA (AutoRegressive Integrated Moving Average)**
3. **Prophet (modelo aditivo de Facebook)**
4. **Random Forest Regressor** con `GridSearchCV` y validación cruzada temporal

##  Evaluación de Modelos

Se aplicaron las siguientes métricas:
- MAE: Mean Absolute Error
- RMSE: Root Mean Squared Error
- MAPE: Mean Absolute Percentage Error

La evaluación incluyó entrenamiento con datos de 1961–2018 y validación con 2019–2023. También se realizó validación extendida con otro particionado.

Gráficos de comparación:
- `evaluacion_modelos_panama.png`
- `evaluacion_modelos_mundial.png`

##  Proyecciones

- **Pronóstico nacional**: superficie agrícola en Panamá (2024–2026 y 2024–2028)
- **Pronóstico por producto**: top 10 productos agrícolas del país
- **Pronóstico global**: evolución total mundial como referencia

Gráficos:
- `pronostico_panama.png`
- `pronostico_consolidado_10_productos_panama.png`
- `pronostico_mundial.png`

## ⚙ Herramientas

- Python 3.10
- pandas, matplotlib, seaborn, scikit-learn, statsmodels, prophet
- Jupyter Notebooks / PyCharm

##  Estructura del Proyecto
