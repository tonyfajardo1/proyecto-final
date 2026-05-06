# Proyecto Final: Predicción de Precios (End-to-End ML con Big Data)

Integrantes: Anthony Fajardo, Nicolas Soria, Ryan De La Torre, Juan Pablo Bautista

> [!WARNING]
> **Arquitectura para Grandes Volúmenes (Big Data):** 
> Procesamiento de **~20GB** de información de viajes. Tratar de descargar toda la data e intentar usar un `train_test_split` tradicional en Pandas o Scikit-Learn saturará su memoria RAM al instante. Por tanto, el corazón de la limpieza estructurada, el ensamble de la OBT y la división Train/Test **se ejecutará del lado de Snowflake mediante SQL**. 

El proyecto está segmentado de la siguiente manera para su evaluación:

*   **PSet #4**: Investigación técnica. Creación del documento técnico (monografía) y exposición sobre el algoritmo de Boosting que se le asignó a su equipo. Algoritmos en paralelo.
*   **Proyecto Final (PSet #5)**: Implementación, experimentación y despliegue del modelo punto a punto.

---

## Objetivo General del Proyecto Final

Deberán conectarse a **Snowflake** para consumir la One Big Table (`analytics.obt_trips`), empujando el cómputo primario a la base de datos (Pushdown Computation), realizar el proceso de de exploración en muestras, estructurar el modelado (*Out-of-Core Training* o por *Lotes*), empaquetarlo en código productivo modular Python y servirlo mediante una API base con FastAPI.

---

## Flujo de Trabajo

### 1. Modelado de Datos
En la subcarpeta `src/data/sql/` estructurarán la lógica de cruce masivo.
1. Script para materializar la OBT unificada.
2. Script para separar los datos (`train_set` 2015-2023, `val_set` 2024, `test_set` 2025).

### 2. Preparación y Exploración
No carguen toda la base. Usen directivas SQL como `SAMPLE` o `LIMIT` mientras evalúan.
1.  **`01_eda.ipynb`**: Realicen Análisis Exploratorio en un **sample**. Identifiquen outliers y Data Leakage.
2.  **`02_data_cleaning.ipynb`**: Validen reglas lógicas en pandas y **traspasen su código estructural a sus queries SQL** en la DB.
3.  **`03_feature_engineering.ipynb`**: Creen variables complejas espacio-temporales.

### 3. Experimentación (Out-of-Core)
**`04_model_experimentation.ipynb`**: Entrenen modelos. Para ensambles y boostings, investiguen sobre la iteración por lotes (`batch training`, iteradores en XGBoost/LightGBM) o tomen la mayor submuestra representativa que soporte la memoria de sus máquinas. Seleccionen el mejor según RMSE.

### 4. Refactorización de Produccion
Migrar el Jupyter a los scripts definitivos en `src/`.
1.  Copiar la lógica de recolección de *chunks* a `src/data/ingestion.py`.
2.  Pipeline definitivo en `src/features/`.
3.  Lógica de `partial_fit` / batch en `src/models/train_model.py`.

### 5. API y Front End
El producto no es un Jupyter, es un software que usará un usuario final interactivo.
1. **Back-end de ML**: Levantar la aplicación web que envuelve al `.pkl` ejecutando:  
   `uvicorn src.api.main:app --reload`
2. **Interfaz de Usuario**: Desarrollar en `app/frontend.py` la interfaz gráfica usando **Streamlit**. El usuario final introducirá datos básicos del viaje y este conectará a la API.  
   Para correr el servidor web, asegúrese de estar en la raíz de su terminal y ejecutar:
   `streamlit run app/frontend.py`

---

## Estado Actual de la Implementacion

El proyecto ya cuenta con el flujo end-to-end implementado:

1. La OBT de modelado se materializa en Snowflake como `ANALYTICS.OBT_TRIPS_MODEL`.
2. Los splits temporales se crean en Snowflake: `ANALYTICS.TRAIN_SET` para 2015-2023, `ANALYTICS.VAL_SET` para 2024 y `ANALYTICS.TEST_SET` para 2025.
3. Los notebooks `01_eda.ipynb`, `02_data_cleaning.ipynb`, `03_feature_engineering.ipynb` y `04_model_experimentation.ipynb` trabajan con muestras para evitar cargar toda la tabla localmente.
4. El mejor modelo seleccionado por RMSE de validacion fue `gradient_boosting`.
5. El modelo productivo fue exportado en `data/processed/price_model.pkl`.
6. La API FastAPI y el frontend Streamlit consumen ese modelo real.

El archivo `data/processed/price_model.pkl` se conserva como artefacto liviano de despliegue para que la API pueda iniciar sin reentrenar. La data cruda y muestras pesadas siguen excluidas del repositorio.

Metricas principales del experimento:

| Split | RMSE | MAE | R2 |
| :--- | ---: | ---: | ---: |
| Validacion 2024 | 6.3425 | 3.9071 | 0.9143 |
| Test sample 2025 | 7.9258 | 4.8191 | 0.8503 |

Modelos comparados:

- Baseline Ridge.
- Random Forest.
- Extra Trees.
- Voting Regressor.
- Bagging.
- Pasting.
- AdaBoost.
- Gradient Boosting.
- XGBoost.
- LightGBM.
- CatBoost.

## Como Ejecutar el Proyecto

Desde la raiz del proyecto:

```powershell
git clone https://github.com/tonyfajardo1/proyecto-final.git
cd proyecto-final
```

Validar tablas de Snowflake:

```powershell
python -m src.data.run_snowflake_setup
```

Reentrenar y exportar el modelo ganador:

```powershell
python -m src.models.train_model --model gradient_boosting
```

Reentrenar y evaluar TEST por lotes desde Snowflake:

```powershell
python -m src.models.train_model --model gradient_boosting --batch-test
```

Ejecutar pruebas:

```powershell
python -m pytest -q
```

Levantar la API:

```powershell
uvicorn src.api.main:app --reload
```

Probar salud de la API:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health"
```

Levantar el frontend:

```powershell
streamlit run app/frontend.py
```

URLs locales:

- API: `http://127.0.0.1:8000`
- Documentacion interactiva: `http://127.0.0.1:8000/docs`
- Frontend: `http://127.0.0.1:8501`

Para la demo en `/docs`, usar primero `POST /predict-simple`. Ese endpoint recibe campos de usuario final como fecha, hora, distancia, pasajeros, boroughs y location IDs. El endpoint `POST /predict` queda disponible como contrato tecnico completo del modelo.

Ejemplo para `POST /predict-simple`:

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

## Orden Recomendado para la Demo

1. Mostrar que Snowflake contiene la OBT y los splits temporales.
2. Abrir los notebooks y explicar que trabajan con muestras, no con toda la tabla en memoria.
3. Mostrar la comparacion de modelos en `04_model_experimentation.ipynb`.
4. Explicar por que `gradient_boosting` fue elegido con `VAL_SET`.
5. Mostrar que `TEST_SET` se usa despues de escoger el ganador.
6. Ejecutar `python -m pytest -q`.
7. Abrir `http://127.0.0.1:8000/docs` y probar `/predict-simple`.
8. Abrir `http://127.0.0.1:8501` y hacer una prediccion desde la interfaz.

---

## Estructura

```text
├── data/               # Archivos prohibidos en Git (.gitignore) y modelos (.pkl)
├── notebooks/          # Exploración interactivo (usar MUESTRAS)
│   ├── 01_eda.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_model_experimentation.ipynb
├── src/                # Código fuente de Producción
│   ├── data/           
│   │   ├── sql/        # Scripts SQL obligatorios para la DB (Pushdown)
│   │   └── ingestion.py # Iterador de descargas
│   ├── features/       # Transformadores sklearn
│   ├── models/         # Entrenamiento modular y por batch
│   ├── api/            # API del modelo (FastAPI)
│   └── utils/          
├── app/                # Carpeta para el Frontend final
│   └── frontend.py     # Aplicación interactiva en Streamlit
├── tests/              # Pruebas unitarias
├── .env.example        
├── requirements.txt    
└── README.md           
```

---
