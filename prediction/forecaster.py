
import logging
import numpy as np
import pandas as pd

from evaluation.metrics import empirical_ic, empirical_coverage
from models import MODEL_REGISTRY

logger = logging.getLogger(__name__)


class Forecaster:
    """
    Genera predicciones futuras usando el modelo ganador por serie.

    Parámetros
    ──────────
    cfg        : dict completo del config.yaml
    df_winners : DataFrame con columnas [idTerminal, oper, mejor_modelo]
    """

    def __init__(self, cfg: dict, df_winners: pd.DataFrame):
        self.cfg        = cfg
        self.df_winners = df_winners
        self.future_idx = pd.date_range(
            cfg["split"]["future_start"],
            cfg["split"]["future_end"],
            freq="D",
        )
        self.n_future = len(self.future_idx)
        self.alpha    = cfg["evaluation"]["alpha"]
        self.min_fac  = cfg["evaluation"]["cv_min_factor"]

    def run(
        self,
        daily    : pd.DataFrame,
        cutoff   : pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Parámetros
        ──────────
        daily  : DataFrame diario completo [fecha_dia, idTerminal, oper, n_tx]
        cutoff : fecha de corte train/test

        Retorna
        ───────
        DataFrame con columnas:
        [idTerminal, oper, ds, yhat, yhat_lower, yhat_upper, modelo_ganador]
        """
        records = []

        for _, row in self.df_winners.iterrows():
            term      = row["idTerminal"]
            op        = row["oper"]
            best_name = row["mejor_modelo"]

            sub = (
                daily[
                    (daily["idTerminal"] == term) &
                    (daily["oper"]       == op)
                ]
                .sort_values("fecha_dia")
            )
            # Para el pronóstico futuro usamos TODA la serie como entrenamiento
            y_full = sub["n_tx"].values

            model_cls = MODEL_REGISTRY[best_name]
            model_cfg = self.cfg["models"][best_name.lower()
                                           if best_name != "HoltWinters"
                                           else "holtwinters"]

            # IC empírico (solo sobre el tramo de entrenamiento pre-cutoff)
            y_train = sub[sub["fecha_dia"] < cutoff]["n_tx"].values
            q_low, q_high, n_folds = empirical_ic(
                y_train, self.n_future,
                model_cls, model_cfg,
                alpha=self.alpha, min_factor=self.min_fac,
            )

            # Ajuste final sobre toda la serie
            try:
                model = model_cls(model_cfg).fit(y_full)
                pred, lower, upper = model.forecast_with_ic(
                    self.n_future, q_low, q_high
                )
            except Exception as e:
                logger.warning(
                    "Forecaster falló T%d/Op%d (%s): %s — usando fallback.",
                    term, op, best_name, e,
                )
                fallback = int(np.mean(y_full[-7:]))
                pred  = np.full(self.n_future, fallback)
                lower = np.maximum(pred - fallback, 0)
                upper = pred + fallback

            for k, d in enumerate(self.future_idx):
                records.append({
                    "idTerminal"    : term,
                    "oper"          : op,
                    "ds"            : d,
                    "yhat"          : pred[k],
                    "yhat_lower"    : lower[k],
                    "yhat_upper"    : upper[k],
                    "modelo_ganador": best_name,
                })

            logger.info(
                "Pronóstico T%d/Op%d → modelo=%s | sum_15d=%d tx",
                term, op, best_name, int(pred.sum()),
            )

        return pd.DataFrame(records)