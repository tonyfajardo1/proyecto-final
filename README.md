
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
