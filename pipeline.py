"""
pipeline.py
----------------------------------------------------------
Estrategia de rendimiento:
  - Modelos activos configurables via config: models.active
  - Rolling CV solo para HoltWinters (modelo ganador dominante).
  - joblib Parallel con n_jobs=16.
  - _fit_one_combination y _forecast_one_row al nivel del modulo (picklables).
"""

import logging, os, time, warnings
import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

from data.ingestion        import DataIngestion
from data.features         import FeatureBuilder
from models                import MODEL_REGISTRY
from evaluation.metrics    import compute_metrics, empirical_ic, empirical_coverage
from evaluation.selector   import ModelSelector
from prediction.forecaster import Forecaster
from visualization.plots   import Plotter

# Solo HW usa rolling CV para IC
CV_IC_MODEL = "HoltWinters"


def setup_logging(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO, format=fmt,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(out_dir, "pipeline.log"), mode="w"),
        ],
    )


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


# ── Funcion de ajuste — nivel de modulo (picklable para joblib) ───────
def _fit_one_combination(term, op, daily, cfg, cutoff_str, n_test, active_models):
    warnings.filterwarnings("ignore")

    CUTOFF    = pd.Timestamp(cutoff_str)
    alpha     = cfg["evaluation"]["alpha"]
    minfac    = cfg["evaluation"]["cv_min_factor"]
    max_folds = cfg["evaluation"].get("max_cv_folds", None)

    sub = (
        daily[(daily["idTerminal"] == term) & (daily["oper"] == op)]
        .sort_values("fecha_dia").reset_index(drop=True)
    )
    train_y  = sub[sub["fecha_dia"] <  CUTOFF]["n_tx"].values
    test_df  = sub[sub["fecha_dia"] >= CUTOFF]
    test_y   = test_df["n_tx"].values
    test_idx = pd.to_datetime(test_df["fecha_dia"].values)

    metrics_rows, preds_rows = [], []
    hw_result = None

    for model_name in active_models:
        model_cls = MODEL_REGISTRY[model_name]
        cfg_key   = {"HoltWinters": "holtwinters",
                     "Prophet"    : "prophet",
                     "SARIMA"     : "sarima"}[model_name]
        model_cfg = cfg["models"][cfg_key]

        try:
            model = model_cls(model_cfg).fit(train_y)
            pred  = model.predict(n_test)
        except Exception:
            pred = np.full(n_test, max(1, int(np.mean(train_y[-7:]))))

        metrics_rows.append(compute_metrics(
            test_y, pred[:len(test_y)],
            model_name=model_name, terminal=term, oper=op,
        ))
        for k, idx in enumerate(test_idx):
            preds_rows.append({
                "idTerminal": term, "oper": op, "ds": idx,
                "y_real": int(test_y[k]),
                "y_pred": int(pred[k]) if k < len(pred) else 0,
                "modelo": model_name,
            })

        # IC empirico rolling CV solo para HoltWinters
        if model_name == CV_IC_MODEL:
            q_low, q_high, n_folds = empirical_ic(
                train_y, n_test, model_cls, model_cfg,
                alpha=alpha, min_factor=minfac, max_folds=max_folds,
            )
            fitted = model_cls(model_cfg).fit(train_y)
            ic_pred, ic_lower, ic_upper = fitted.forecast_with_ic(
                n_test, q_low, q_high
            )
            cob    = empirical_coverage(test_y, ic_lower, ic_upper)
            mae_ic = float(np.mean(np.abs(test_y - ic_pred[:len(test_y)])))
            hw_result = dict(
                pred=ic_pred, lower=ic_lower, upper=ic_upper,
                test_y=test_y, test_idx=test_idx,
                mae=mae_ic, cobertura=cob, n_folds=n_folds,
            )

    return dict(term=term, op=op,
                metrics_rows=metrics_rows,
                preds_rows=preds_rows,
                hw_result=hw_result)


# ── Funcion de pronostico — nivel de modulo (picklable para joblib) ───
def _forecast_one_row(row, daily, cfg, cutoff):
    import warnings
    warnings.filterwarnings("ignore")
    import pandas as pd
    from prediction.forecaster import Forecaster

    fc = Forecaster(cfg, pd.DataFrame([row]))
    return fc.run(daily, cutoff)


# ── Pipeline principal ────────────────────────────────────────────────
def run_pipeline(config_path):
    cfg     = load_config(config_path)
    out_dir = cfg["paths"]["output_dir"]
    setup_logging(out_dir)
    logger  = logging.getLogger("pipeline")

    logger.info("=" * 60)
    logger.info("PIPELINE: %s  v%s", cfg["project"]["name"], cfg["project"]["version"])
    logger.info("=" * 60)
    t0 = time.time()

    active_models = cfg["models"].get("active", ["HoltWinters", "Prophet"])
    logger.info("Modelos activos: %s", active_models)

    # PASO 1 — Ingesta
    logger.info("--- PASO 1: Ingesta ---")
    df_raw = DataIngestion(cfg).run()

    # PASO 2 — Features
    logger.info("--- PASO 2: Feature engineering ---")
    fb        = FeatureBuilder(cfg)
    daily     = fb.run(df_raw)
    terminals = fb.top_terminals
    opers     = fb.top_operations

    # PASO 3 — EDA
    logger.info("--- PASO 3: EDA ---")
    plotter = Plotter(cfg, out_dir)
    plotter.plot_eda_overview(df_raw)
    plotter.plot_temporal_patterns(df_raw)
    plotter.plot_series_grid(daily, terminals, opers)

    # PASO 4 — Ajuste paralelo
    N_TEST       = 16
    cutoff_str   = cfg["split"]["cutoff_date"]
    combinations = [(t, o) for t in terminals for o in opers]
    logger.info("--- PASO 4: Ajuste paralelo | %d combinaciones x %d modelos ---",
                len(combinations), len(active_models))

    results = Parallel(n_jobs=16, verbose=3)(
        delayed(_fit_one_combination)(
            t, o, daily, cfg, cutoff_str, N_TEST, active_models
        )
        for t, o in combinations
    )

    all_metrics, all_preds, hw_results = [], [], {}
    for r in results:
        all_metrics.extend(r["metrics_rows"])
        all_preds.extend(r["preds_rows"])
        if r["hw_result"] is not None:
            hw_results[(r["term"], r["op"])] = r["hw_result"]
        logger.info("  T%d/Op%d completado", r["term"], r["op"])

    df_metrics = pd.DataFrame(all_metrics)
    df_preds   = pd.DataFrame(all_preds)

    # PASO 5 — Seleccion
    logger.info("--- PASO 5: Seleccion de modelo ganador ---")
    selector   = ModelSelector(metric=cfg["evaluation"]["metric_selector"])
    df_winners = selector.run(df_metrics)
    summary    = selector.summary_table(df_metrics)
    logger.info("Resumen global:\n%s", summary.to_string())

    # PASO 6 — Pronostico paralelizado
    logger.info("--- PASO 6: Pronostico 15 dias (paralelo) ---")
    df_future = pd.concat(
        Parallel(n_jobs=16, verbose=3)(
            delayed(_forecast_one_row)(
                row, daily, cfg, pd.Timestamp(cutoff_str)
            )
            for _, row in df_winners.iterrows()
        ),
        ignore_index=True
    )

    # PASO 7 — Visualizaciones
    logger.info("--- PASO 7: Visualizaciones ---")
    plotter.plot_model_comparison_boxplots(df_metrics)
    plotter.plot_mae_heatmaps(df_metrics)
    plotter.plot_winner_summary(df_winners, df_metrics, opers)
    plotter.plot_holdout_all_models(df_preds, df_winners, terminals, opers)
    plotter.plot_hw_ic95(daily, hw_results, terminals, opers)
    plotter.plot_forecast_winners(daily, df_future, terminals, opers)
    plotter.plot_demand_heatmap(df_future)
    plotter.plot_stl_total(df_raw)
    plotter.plot_stl_5x5(daily, terminals, opers)
    plotter.plot_acf_total(df_raw)
    plotter.plot_acf_5x5(daily, terminals, opers)
    plotter.plot_dow_5x5(daily, terminals, opers)
    plotter.plot_hora_5x5(df_raw, terminals, opers)
    plotter.plot_demand_heatmap_oper(df_future)         
    plotter.plot_demand_heatmap_term_oper(df_future)    
    plotter.plot_demand_participacion(df_future)        

    # PASO 8 — Artefactos
    logger.info("--- PASO 8: Artefactos ---")
    df_metrics.to_csv(os.path.join(out_dir, "metricas_modelos.csv"),    index=False)
    df_winners.to_csv(os.path.join(out_dir, "modelos_ganadores.csv"),   index=False)
    df_future.to_csv(os.path.join(out_dir,  "predicciones_15dias.csv"), index=False)

    logger.info("PIPELINE COMPLETADO en %.1fs | %s", time.time() - t0, out_dir)
    return dict(df_metrics=df_metrics, df_winners=df_winners,
                df_future=df_future, summary=summary)