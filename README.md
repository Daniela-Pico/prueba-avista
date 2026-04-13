# Análisis Predictivo del Volumen Transaccional en Terminales Bancarias

**Avista · Análisis de Datos · Abril 2026**

---

## Descripción

Pipeline MLOps completo para el análisis y predicción del volumen de transacciones
en terminales bancarias. A partir de **775.466 transacciones** registradas entre
enero y mayo de 2017, se construyó un modelo predictivo capaz de estimar la demanda
diaria por operación en cada terminal para los próximos **15 días**.

---

## Resultados principales

| Indicador | Valor |
|---|---|
| Transacciones analizadas | 775.466 |
| Terminales activas | 47 |
| Tipos de operación | 46 |
| Período | Ene–May 2017 (121 días) |
| Modelo ganador | Holt-Winters aditivo estacional |
| MAE (error promedio) | 14.8 transacciones/día |
| Cobertura IC 95% | 88–100% |

---

## Estructura del proyecto

```
prueba-avista/
├── config/
│   └── config.yaml          ← hiperparámetros centralizados
├── data/
│   ├── ingestion.py         ← carga, validación y limpieza
│   └── features.py          ← agregación diaria, relleno de ceros
├── models/
│   ├── base_model.py        ← interfaz abstracta BaseModel
│   ├── holtwinters_model.py ← Holt-Winters hereda BaseModel
│   ├── prophet_model.py     ← Prophet hereda BaseModel
│   └── sarima_model.py      ← SARIMA hereda BaseModel
├── evaluation/
│   ├── metrics.py           ← MAE, RMSE, MAPE + IC empírico rolling CV
│   └── selector.py          ← selección del modelo ganador por serie
├── visualization/
│   └── plots.py             ← 19 figuras, 0 lógica de negocio
├── prediction/
│   └── forecaster.py        ← pronóstico 15 días con IC empírico
├── pipeline.py              ← orquestador principal, joblib.Parallel
├── run.py                   ← punto de entrada CLI
└── informe_avista.html      ← informe ejecutivo completo
```

---

## Informe ejecutivo

El informe completo con análisis exploratorio, modelos, predicciones e interpretaciones
está disponible aquí:

👉 **[Ver informe ejecutivo](https://daniela-pico.github.io/prueba-avista/informe_avista.html)**

---

## Modelos evaluados

| Modelo | MAE | RMSE | MAPE % | Series ganadas |
|---|---|---|---|---|
| **Holt-Winters** ✓ | **14.80** | **19.16** | 64.9% | **17/25** |
| Prophet | 15.36 | 19.22 | 77.7% | 8/25 |
| SARIMA | 15.91 | 20.93 | 74.4% | 4/25 |

Se evaluaron los 3 modelos sobre un **holdout temporal de 15 días**.
El modelo ganador se seleccionó por serie (combinación terminal × operación)
según el menor MAE.

---

## Cómo ejecutar

### Requisitos

```bash
pip install prophet pmdarima statsmodels scikit-learn pyyaml seaborn matplotlib joblib pandas numpy nbformat nbconvert
```

### Ejecutar el pipeline completo

```bash
python run.py
```

### Configuración

Todos los hiperparámetros están centralizados en `config/config.yaml`.
Para activar SARIMA cambiar:

```yaml
models:
  active: ["HoltWinters", "Prophet", "SARIMA"]
```

### Generar el informe HTML

```bash
python exportar.py
```

---

## Hallazgos principales

- La **operación 0** concentra el **54%** del volumen total
- Las **5 terminales más activas** representan el **19%** del total
- Caída del **~40%** en fines de semana vs días hábiles
- Estacionalidad semanal confirmada por STL, ACF y análisis descriptivo
- La terminal **1910** es la más crítica con hasta **465 tx/día** predichas

---

## Tecnologías

| Librería | Uso |
|---|---|
| `pandas` / `numpy` | Manipulación de datos |
| `statsmodels` | Holt-Winters, STL, ACF |
| `prophet` | Modelo Prophet |
| `pmdarima` | SARIMA con auto_arima |
| `scikit-learn` | Métricas de evaluación |
| `joblib` | Paralelización (n_jobs=16) |
| `matplotlib` / `seaborn` | Visualizaciones |
| `pyyaml` | Configuración centralizada |

---

*Daniela Pico Arredondo · Avista · 2026*
