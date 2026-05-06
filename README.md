# Proyecto Final: Predicción de Precios NYC Taxi

Proyecto end-to-end de Machine Learning para predecir el valor total de viajes de taxi de Nueva York usando procesamiento Big Data en Snowflake, experimentación con modelos de ensamble y despliegue mediante FastAPI y Streamlit.

## Integrantes

- Anthony Fajardo
- Nicolas Soria
- Ryan De La Torre
- Juan Pablo Bautista

## Resumen del Proyecto

El objetivo fue construir una solución completa de predicción de precios para viajes de taxi, evitando cargar localmente el volumen completo de datos. El procesamiento principal se realizó en Snowflake mediante SQL, y Python se usó para exploración con muestras, entrenamiento controlado, evaluación, API y frontend.

La solución final incluye:

- Construcción de una One Big Table de modelado en Snowflake.
- Limpieza estructurada y eliminación de variables con riesgo de data leakage.
- Splits temporales creados en base de datos.
- Notebooks de EDA, limpieza, feature engineering y experimentación.
- Comparación de modelos baseline, bagging, pasting, voting y boosting.
- Modelo final exportado como artefacto `.pkl`.
- API REST con FastAPI.
- Interfaz gráfica con Streamlit consumiendo la API.
- Pruebas unitarias para validar features, API y modelo exportado.

## Arquitectura Big Data

El dataset completo contiene cientos de millones de registros, por lo que el flujo evita descargar toda la información en memoria local. La limpieza principal, la construcción de la OBT y los splits se ejecutan del lado de Snowflake.

Objetos principales en Snowflake:

| Objeto | Periodo | Uso |
| :--- | :--- | :--- |
| `ANALYTICS.OBT_TRIPS_MODEL` | 2015-2025 | Tabla final de modelado |
| `ANALYTICS.TRAIN_SET` | 2015-2023 | Entrenamiento |
| `ANALYTICS.VAL_SET` | 2024 | Selección de modelo |
| `ANALYTICS.TEST_SET` | 2025 | Evaluación final |

Volumen validado en los notebooks:

| Objeto | Filas |
| :--- | ---: |
| `OBT_TRIPS_MODEL` | 844,000,986 |
| `TRAIN_SET` | 760,168,027 |
| `VAL_SET` | 39,723,735 |
| `TEST_SET` | 44,109,224 |

Para EDA y experimentación se usan consultas agregadas, `SAMPLE BERNOULLI`, `LIMIT` y evaluación por lotes. No se usa `train_test_split` local porque el problema requiere separación temporal.

## Flujo de Trabajo

### 1. Ingeniería de Datos en Snowflake

Los scripts SQL están en `src/data/sql/`:

- `01_create_obt.sql`: crea `ANALYTICS.OBT_TRIPS_MODEL`.
- `02_create_splits.sql`: crea `TRAIN_SET`, `VAL_SET` y `TEST_SET`.

La OBT conserva solo variables disponibles antes o al inicio del viaje. Se excluyeron columnas posteriores al cierre del viaje, como tarifas desagregadas, propinas, duración real, velocidad promedio y tipo de pago.

### 2. Notebooks

Los notebooks documentan la parte exploratoria y experimental:

| Notebook | Propósito |
| :--- | :--- |
| `01_eda.ipynb` | Análisis exploratorio sobre muestras y agregados de Snowflake |
| `02_data_cleaning.ipynb` | Validación de reglas de limpieza y chequeo de leakage |
| `03_feature_engineering.ipynb` | Construcción y validación de variables del modelo |
| `04_model_experimentation.ipynb` | Comparación de modelos y selección del ganador |

### 3. Modelado

Se compararon modelos lineales, ensambles y boostings:

- Baseline Ridge
- Random Forest
- Extra Trees
- Voting Regressor
- Bagging
- Pasting
- AdaBoost
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost

El modelo seleccionado fue `gradient_boosting`, elegido con `VAL_SET` 2024. El `TEST_SET` 2025 se usó después de seleccionar el ganador.

Métricas principales:

| Split | RMSE | MAE | R2 |
| :--- | ---: | ---: | ---: |
| Validación 2024 | 6.3425 | 3.9071 | 0.9143 |
| Test sample 2025 | 7.9258 | 4.8191 | 0.8503 |

El modelo productivo se exportó en:

```text
data/processed/price_model.pkl
```

Este artefacto se mantiene en el repositorio para que la API y el frontend puedan ejecutarse sin reentrenar.

## Estructura del Repositorio

```text
.
|-- app/
|   `-- frontend.py
|-- data/
|   |-- interim/
|   |-- processed/
|   |   `-- price_model.pkl
|   `-- raw/
|-- notebooks/
|   |-- 01_eda.ipynb
|   |-- 02_data_cleaning.ipynb
|   |-- 03_feature_engineering.ipynb
|   `-- 04_model_experimentation.ipynb
|-- src/
|   |-- api/
|   |   `-- main.py
|   |-- data/
|   |   |-- ingestion.py
|   |   |-- run_snowflake_setup.py
|   |   `-- sql/
|   |       |-- 01_create_obt.sql
|   |       `-- 02_create_splits.sql
|   |-- features/
|   |   `-- build_features.py
|   |-- models/
|   |   |-- predict_model.py
|   |   `-- train_model.py
|   `-- utils/
|       `-- config.py
|-- tests/
|-- .env.example
|-- .gitignore
|-- README.md
`-- requirements.txt
```

## Instalación

Clonar el repositorio:

```powershell
git clone https://github.com/tonyfajardo1/proyecto-final.git
cd proyecto-final
```

Crear y activar un entorno virtual:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Instalar dependencias:

```powershell
pip install -r requirements.txt
```

Crear archivo de variables de entorno:

```powershell
copy .env.example .env
```

Luego completar `.env` con las credenciales reales de Snowflake.

## Ejecución del Proyecto

Validar y crear objetos en Snowflake:

```powershell
python -m src.data.run_snowflake_setup
```

Reentrenar el modelo ganador:

```powershell
python -m src.models.train_model --model gradient_boosting
```

Evaluar el `TEST_SET` por lotes desde Snowflake:

```powershell
python -m src.models.train_model --model gradient_boosting --batch-test
```

Ejecutar pruebas:

```powershell
python -m pytest -q
```

Resultado esperado:

```text
5 passed
```

## API FastAPI

Levantar la API:

```powershell
uvicorn src.api.main:app --reload
```

URLs:

- API: `http://127.0.0.1:8000`
- Documentación interactiva: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

Endpoint recomendado para la demo:

```text
POST /predict-simple
```

Ejemplo de entrada:

```json
{
  "pickup_date": "2025-01-15",
  "pickup_time": "12:00:00",
  "trip_distance": 2.5,
  "passenger_count": 1,
  "service_type": "yellow",
  "vendor_id": "2",
  "rate_code_id": "1",
  "trip_type": "1",
  "pu_borough": "Manhattan",
  "do_borough": "Manhattan",
  "pu_location_id": 237,
  "do_location_id": 236
}
```

El endpoint `/predict-simple` recibe campos cercanos a lo que ingresaría un usuario final. El endpoint `/predict` queda disponible como contrato técnico completo del modelo.

## Frontend Streamlit

Levantar la interfaz:

```powershell
streamlit run app/frontend.py
```

URL local:

```text
http://127.0.0.1:8501
```

La interfaz permite ingresar datos básicos del viaje y consultar la API para obtener la tarifa estimada.

## Control de Data Leakage

El proyecto controla el leakage en tres niveles:

- La OBT excluye variables conocidas después del viaje, como `FARE_AMOUNT`, `TIP_AMOUNT`, `TOLLS_AMOUNT`, `PAYMENT_TYPE`, `TRIP_DURATION_MIN` y `AVG_SPEED_MPH`.
- Los splits se hacen por tiempo en Snowflake, no con partición aleatoria local.
- El `TEST_SET` 2025 se reserva para evaluación posterior a la selección del modelo ganador.

## Orden Sugerido para la Demo

1. Mostrar los scripts SQL de la OBT y los splits temporales.
2. Abrir los notebooks y explicar que usan muestras, no la tabla completa.
3. Mostrar el resumen de filas de Snowflake.
4. Mostrar la comparación de modelos en `04_model_experimentation.ipynb`.
5. Explicar la selección de `gradient_boosting` con validación 2024.
6. Ejecutar `python -m pytest -q`.
7. Probar `/predict-simple` en `http://127.0.0.1:8000/docs`.
8. Probar una predicción desde Streamlit.

## Notas de Reproducibilidad

- `.env` no se versiona porque contiene credenciales.
- `.env.example` documenta las variables necesarias.
- Los datos crudos e intermedios no se suben al repositorio.
- El modelo final `price_model.pkl` se conserva como artefacto liviano de despliegue.
- La ejecución completa depende de tener acceso válido a Snowflake y a las tablas fuente del proyecto.
