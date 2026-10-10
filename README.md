# Competencia 1 — Predicción de churn bancario (BAJA+2, agosto 2021) — DMEyF 2026

Autora: Micaela R. Rodriguez

Este repositorio permite replicar **exactamente** el archivo enviado como entrega final:

| | |
|---|---|
| Archivo final | `prediccion_final/prediccion_baja_202108_lgbm_binario_v11_corte11500_20261009_224353.csv` |
| Contenido | 11.500 `numero_de_cliente` (una sola columna, sin encabezado, enteros) |
| SHA-256 | `2d690cb36f34445efa9a4b81726037c2bcba30c88c164e6d249137ec37d81b0a` |

## 1. Estructura del repositorio

```
├── README.md
├── prediccion_final/
│   └── prediccion_baja_202108_lgbm_binario_v11_corte11500_20261009_224353.csv   <- el submit final
├── 01_prediccion_final/
│   └── churn_pipeline_final.ipynb          <- UN notebook: competencia_01.csv -> CSV final
└── 02_optimizacion_bayesiana/              <- de dónde salen los hiperparámetros (ver sección 4)
    ├── configs_finales.json                <- las 6 configuraciones usadas en el ensamble
    ├── optuna_lgbm_binario_v9.db           <- estudio de Optuna completo (60 trials)
    ├── churn_pipeline_v9_optuna.ipynb      <- corrida que generó los trials 0–36
    ├── churn_pipeline_v11_corrida_original.ipynb   <- corrida que generó los trials 37–61 y el submit (con salidas)
    ├── resumen_cortes_corrida_original.csv
    └── origen_warm_start/                  <- estudios previos que sembraron el punto de partida de Optuna
```

## 2. Resumen de la solución

- **Datos:** `competencia_01.csv` (202103–202108). Se predice BAJA+2 en 202108.
- **Meses:** entrenamiento 202103–05 y validación 202106 (para decidir); entrenamiento final 202103–06; aplicación 202108. 202107 no tiene target y se usa sólo para lags e historia.
- **Target:** binario (BAJA+1 y BAJA+2 = 1). La ganancia se mide sólo sobre BAJA+2 (+1.072.500 por acierto, −27.500 por error).
- **Sin undersampling** en ninguna etapa (hay un `assert`).
- **Calidad de datos:** `ccajas_depositos` en NA para 202105 (todos ceros). Montos deflactados por IPC (los límites de tarjeta se conservan también en nominal). Se excluyen `mpayroll`, `cpayroll_trx`, `mprestamos_personales`, `cprestamos_personales`, sus lags/deltas y sus derivadas (hoja de ruta).
- **Drift:** se excluyen las variables con PSI > 0,25 (train vs. 202108), con sus lags/deltas. En la entrega, además, se sacan los límites de tarjeta nominales y `mcomisiones_mantenimiento`.
- **Features:** lags y deltas de 15 variables, `delta1` de todas las numéricas, ratios/agrupaciones, cluster KMeans (k=5, ajustado sólo en train) como dummies y **175 features de historia** (media, desvío, mínimo, máximo, desvío relativo, apartamiento contra la media y racha de caídas sobre los últimos 3 meses) de 25 variables. Total: **504 columnas**.
- **Pesos por mes:** cada fila pesa `0,8 ** (meses desde el último mes de entrenamiento)`.
- **Modelo:** LightGBM binario. **Ensamble de 6 configuraciones × 2 semillas = 12 modelos**, entrenados con 202103–202106, y se **promedian las probabilidades por cliente**.
- **Corte:** los 11.500 clientes con mayor probabilidad (ver sección 5).

## 3. Cómo replicar el archivo final

1. Poner `competencia_01.csv` en la carpeta del notebook o en una carpeta superior (hasta 3 niveles), o definir `CHURN_BASE_PATH` con la carpeta que lo contiene. El notebook crea ahí `duckdb_tmp/` (se borra solo) y `salida_final_<fecha>/`.
2. Instalar el entorno (sección 6).
3. Ejecutar `01_prediccion_final/churn_pipeline_final.ipynb` de principio a fin (*Run All*). No requiere nada más: las 6 configuraciones de hiperparámetros, las 25 variables con historia, el decay (0,8), el conjunto de columnas y el corte (11.500) están fijados en el notebook.
4. El archivo sale en `salida_final_<fecha>/prediccion_baja_202108_lgbm_binario_final_corte11500_<fecha>.csv`, junto con `meta.json` (versiones de librerías, cantidad de CPU, configuraciones) y `resumen_cortes.csv`. El nombre lleva la fecha de la corrida; el **contenido** debe coincidir con el archivo de `prediccion_final/`.
5. Verificación: el notebook comprueba que quedan 504 columnas; comparar `sha256sum` del CSV generado con el de la tabla de arriba, o comparar el conjunto de IDs.

Tiempo: unos 40 minutos de entrenamiento (≈18 min el ensamble en validación y ≈21 min el entrenamiento final, medidos en la corrida original) más el ETL y las features. La corrida original usó una máquina Windows con ~34 GB de RAM; se recomiendan al menos 16 GB (DuckDB trabaja en disco con límite de memoria del 40% de la RAM).

**Sobre la reproducibilidad exacta:** todas las semillas están fijadas, pero LightGBM usa `num_threads = os.cpu_count()` y la suma en punto flotante con distinta cantidad de hilos o distinta versión de la librería puede alterar los últimos decimales de las probabilidades. Si el archivo generado no coincide bit a bit, esperar diferencias sólo en los clientes cercanos al corte. Por eso `meta.json` registra el entorno de cada corrida. Por la misma razón las 25 variables con historia **están escritas en el notebook** (en la corrida original salieron de un ranking de importancia que cambió entre dos corridas).

## 4. Procedencia de los hiperparámetros (optimización bayesiana)

Los hiperparámetros del ensamble **no se optimizan dentro del notebook final**: salen de un estudio de Optuna que se guardó y se incluye completo.

- **Estudio:** `02_optimizacion_bayesiana/optuna_lgbm_binario_v9.db` (nombre de estudio `lgbm_binario`; 60 trials completos y 2 fallidos por interrupción manual).
- **Objetivo de Optuna:** máximo de la curva de ganancia suavizada (media móvil de 401 envíos) en 202106, entrenando con 202103–05, **sin undersampling**, pesos por mes 0,8 y 3 semillas por trial. Sampler TPE (semilla 17).
- **Espacio de búsqueda:** `learning_rate` 0,01–0,1 (log), `num_leaves` 8–1024 (log), `min_data_in_leaf` 50–5000 (log), `feature_fraction` 0,1–0,8, `bagging_fraction` 0,5–1, `lambda_l2` 0,001–100 (log), `num_boost_round` 100–2000 (log).
- **Cómo se corrió:** el estudio se fue continuando en el tiempo (límite de 3–4 h por corrida): los trials 0–36 los generó `churn_pipeline_v9_optuna.ipynb` (8/10/2026) y los trials 37–61, `churn_pipeline_v11_corrida_original.ipynb` (9/10/2026). Los trials 0–36 se evaluaron con 502 columnas y los siguientes con 504 (las dos corridas difirieron en una de las 25 variables con historia).
- **Punto de partida:** los trials 0–4 son la reevaluación de los 5 mejores parámetros de estudios previos (`origen_warm_start/`: `optuna_lgbm_binario_v5.db`, con trials de las corridas v5 y v7, y `optuna_lgbm_binario_v3.db`). Ninguna de las 6 configuraciones finales es un trial de partida; sí influyeron en el muestreo de TPE. Se incluyen los notebooks y bases de esos estudios por completitud.

**Las 6 configuraciones del ensamble** (las mejores 6 del estudio por valor de Optuna; también en `configs_finales.json` y en la celda de configuraciones del notebook final):

| Trial | Corrida | Valor Optuna (202106) | lr | hojas | min_data_in_leaf | feature_fraction | bagging_fraction | lambda_l2 | rondas |
|---|---|---|---|---|---|---|---|---|---|
| 54 | v11 | 486,49 M | 0,01561 | 27 | 766 | 0,15445 | 0,99476 | 0,37874 | 484 |
| 32 | v9 | 480,29 M | 0,01453 | 43 | 1736 | 0,15713 | 0,96926 | 1,49430 | 631 |
| 23 | v9 | 478,29 M | 0,01724 | 26 | 945 | 0,10311 | 0,89063 | 0,49242 | 516 |
| 26 | v9 | 478,22 M | 0,01549 | 78 | 3052 | 0,10833 | 0,78959 | 0,84924 | 571 |
| 43 | v11 | 476,87 M | 0,01922 | 44 | 1418 | 0,10237 | 0,96525 | 0,82816 | 501 |
| 31 | v9 | 476,51 M | 0,01331 | 28 | 1399 | 0,13433 | 0,87958 | 0,78651 | 624 |

(Valores completos en `configs_finales.json`.)

**Qué hacer según el caso:**
- *Replicar la predicción final:* nada más que la sección 3; los hiperparámetros ya están en el notebook.
- *Verificar que las configuraciones salen del estudio:* abrir `optuna_lgbm_binario_v9.db` (`optuna.load_study(study_name="lgbm_binario", storage="sqlite:///optuna_lgbm_binario_v9.db")`) y tomar los 6 mejores trials completos; deben coincidir con la tabla.
- *Volver a correr la optimización:* correr `churn_pipeline_v9_optuna.ipynb` (copiando `origen_warm_start/*.db` a una carpeta `modelos/` junto al dataset) y luego `churn_pipeline_v11_corrida_original.ipynb`, que retoma el mismo estudio. No se espera obtener los mismos trials: la búsqueda estuvo limitada por tiempo y se interrumpió y retomó varias veces.

Observación: el valor de Optuna de los mejores trials es optimista respecto del real (se elige el máximo sobre un solo mes de validación): al reentrenar las 6 configuraciones con 2 semillas, la ganancia en 202106 de cada una difiere entre −9 y +12 BAJA+2 del valor de Optuna. Por eso se usa el ensamble y no el mejor trial solo.

## 5. Elección del corte (11.500 envíos)

Los cortes de 10.000 a 14.500 (en pasos de 500) salen del **mismo ranking** (sólo cambia cuántos clientes se contactan). Se enviaron varios al pseudo-Kaggle (leaderboard público) y **el mejor resultado fue el de 11.500**, que es el que se entrega.

Como referencia, la ganancia suavizada del ensamble en 202106 (cortes de agosto reescalados por la cantidad de clientes del mes):

| Envíos en agosto | Equivale en 202106 | Ganancia suavizada |
|---|---|---|
| 10.000 | 9.968 | 458,9 M |
| 11.000 | 10.964 | 468,3 M |
| **11.500 (elegido)** | 11.463 | **472,9 M** |
| 12.000 | 11.961 | 475,0 M |
| 12.500 | 12.460 | 480,6 M |
| 12.720 (óptimo en validación) | 12.679 | 482,5 M |
| 13.000 | 12.958 | 480,1 M |
| 14.000 | 13.955 | 472,6 M |

La validación en 202106 favorece un corte mayor (óptimo en 12.720; 11.500 queda unos 9,6 M, unos 9 BAJA+2, por debajo), mientras que el pseudo-Kaggle de agosto favoreció 11.500. El leaderboard público es una muestra ruidosa (una diferencia de ~4% equivale a unos 17 BAJA+2), así que la elección se apoya en la evidencia disponible sobre agosto, que es el mes que se evalúa.

## 6. Entorno

- Python 3.13 (Anaconda), Windows 11. Paquetes: `duckdb`, `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `optuna`, `matplotlib`, `psutil` (opcional; fija el límite de memoria de DuckDB).
- Versiones usadas en la corrida original: `<completar con las del entorno de la entrega: pip list | findstr "lightgbm duckdb pandas numpy scikit-learn optuna">`.
- El notebook final guarda las versiones y la cantidad de CPU en `meta.json`.

## 7. Qué se probó y se descartó (resumen)

Todas las comparaciones se hicieron sobre 202106 con un semillero de 10 modelos; una diferencia de 1–3 BAJA+2 se considera ruido.

- **Random Forest** (ternario y binario): unos 40 BAJA+2 por debajo de LightGBM y 30–100 veces más lento.
- **Undersampling de CONTINUA:** descartado por decisión (pocos meses; recomendación de la cátedra).
- **Recorte de variables con "canaritos"** (variables aleatorias de control): −24 a −36 BAJA+2.
- **Sacar variables con errores de datos y variables de fecha** (`cliente_vip`, conteos de préstamos hipotecarios/prendarios, `*_descuentos`, `ccajas_depositos`, `*_Fvencimiento`, `*_Finiciomora`, `*_fultimo_cierre`, `*_fechaalta`): −2 a −8 BAJA+2; las fechas aportan señal, así que se conservan.
- **Receta de hiperparámetros de la cátedra** (`lr 0,005`, `max_bin 31`, `min_data_in_leaf 0`, `feature_fraction 0,5`, `feature_fraction_bynode 0,2`) con una configuración inicial: −21 BAJA+2 y 4 veces más lenta; el estudio de Optuna correspondiente no llegó a completar trials, por lo que no es concluyente.
- **Features de historia:** aportan poco por sí solas (+0,65 BAJA+2 sobre las mismas columnas sin historia, dentro del ruido); se mantienen en la entrega porque así se entrenó el submit.
- **Lo que sí mejoró:** pesos por mes (decay 0,8: +3,4 BAJA+2), ensamble de configuraciones, límites de tarjeta sólo en nominal (evita un drift artificial) y sacar `mcomisiones_mantenimiento`.
