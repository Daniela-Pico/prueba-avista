import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """
    Construye las series de tiempo diarias para modelado.

    Parámetros
    ──────────
    cfg : dict — sección 'data' + 'split' del config.yaml
    """

    def __init__(self, cfg: dict):
        self.top_n_term  = cfg["data"]["top_n_terminals"]
        self.top_n_oper  = cfg["data"]["top_n_opers"]
        self.term_col    = cfg["data"]["terminal_col"]
        self.oper_col    = cfg["data"]["oper_col"]

        # Se populan en run()
        self.top_terminals  : list = []
        self.top_operations : list = []
        self.all_days       : pd.DatetimeIndex = None

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parámetros
        ──────────
        df : DataFrame limpio proveniente de DataIngestion.run()

        Retorna
        ───────
        pd.DataFrame con columnas [fecha_dia, idTerminal, oper, n_tx]
        completo (sin gaps de fechas), con n_tx = 0 donde no hubo transacciones.
        """
        self._select_entities(df)
        df_sel  = self._filter(df)
        daily   = self._aggregate(df_sel)
        daily   = self._fill_gaps(daily, df)
        logger.info(
            "Series construidas: %d filas | %d combinaciones (terminal×oper)",
            len(daily),
            daily.groupby([self.term_col, self.oper_col]).ngroups,
        )
        return daily

    # ── Pasos internos ────────────────────────────────────────────────────

    def _select_entities(self, df: pd.DataFrame):
        """
        Criterio: top-N por volumen total.
        Justificación: mayor representatividad estadística y
        mayor impacto operativo en el negocio.
        """
        self.top_terminals  = (
            df[self.term_col].value_counts()
            .head(self.top_n_term).index.tolist()
        )
        self.top_operations = (
            df[self.oper_col].value_counts()
            .head(self.top_n_oper).index.tolist()
        )
        logger.info("Terminales seleccionadas : %s", self.top_terminals)
        logger.info("Operaciones seleccionadas: %s", self.top_operations)

    def _filter(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = (
            df[self.term_col].isin(self.top_terminals) &
            df[self.oper_col].isin(self.top_operations)
        )
        df_sel = df[mask].copy()
        logger.info(
            "Filtro aplicado: %d → %d registros (%.1f%%)",
            len(df), len(df_sel), 100 * len(df_sel) / len(df),
        )
        return df_sel

    def _aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        daily = (
            df.groupby(["fecha_dia", self.term_col, self.oper_col])
            .size()
            .reset_index(name="n_tx")
        )
        return daily

    def _fill_gaps(self, daily: pd.DataFrame, df_full: pd.DataFrame) -> pd.DataFrame:
        """
        Rellena con ceros los días sin transacciones para cada combinación,
        garantizando series de tiempo continuas y de igual longitud.
        """
        self.all_days = pd.date_range(
            df_full["fecha_dia"].min(),
            df_full["fecha_dia"].max(),
            freq="D",
        )
        idx_full = pd.MultiIndex.from_product(
            [self.all_days, self.top_terminals, self.top_operations],
            names=["fecha_dia", self.term_col, self.oper_col],
        )
        daily = (
            daily
            .set_index(["fecha_dia", self.term_col, self.oper_col])
            .reindex(idx_full, fill_value=0)
            .reset_index()
        )
        n_zeros = (daily["n_tx"] == 0).sum()
        logger.info(
            "Gaps rellenados con 0: %d días-combinación (%.1f%%)",
            n_zeros, 100 * n_zeros / len(daily),
        )
        return daily