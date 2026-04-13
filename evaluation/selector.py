
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class ModelSelector:
    """
    Selecciona el mejor modelo por combinación.

    Parámetros
    ──────────
    metric : str — nombre de la columna métrica a minimizar (ej. 'MAE')
    """

    def __init__(self, metric: str = "MAE"):
        self.metric = metric

    def run(self, df_metrics: pd.DataFrame) -> pd.DataFrame:
        """
        Parámetros
        ──────────
        df_metrics : DataFrame con columnas
                     [idTerminal, oper, modelo, MAE, RMSE, MAPE_%]

        Retorna
        ───────
        DataFrame con columnas [idTerminal, oper, mejor_modelo, <metric>]
        """
        idx = (
            df_metrics
            .groupby(["idTerminal", "oper"])[self.metric]
            .idxmin()
        )
        winners = df_metrics.loc[idx, ["idTerminal", "oper", "modelo", self.metric]].copy()
        winners = winners.rename(columns={"modelo": "mejor_modelo"})
        winners = winners.reset_index(drop=True)

        # Log resumen
        counts = winners["mejor_modelo"].value_counts()
        logger.info(
            "Selección de modelos ganadores (%s mínimo):\n%s",
            self.metric,
            counts.to_string(),
        )
        return winners

    def summary_table(self, df_metrics: pd.DataFrame) -> pd.DataFrame:
        """
        Tabla resumen: MAE promedio, mediana y std por modelo.
        Útil para el reporte final.
        """
        return (
            df_metrics
            .groupby("modelo")[["MAE", "RMSE", "MAPE_%"]]
            .agg(["mean", "median", "std"])
            .round(2)
            .sort_values(("MAE", "mean"))
        )