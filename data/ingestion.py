
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class DataIngestion:
    """
    Carga y valida el dataset transaccional.

    Parámetros
    ──────────
    cfg : dict  — sección 'data' + 'paths' del config.yaml
    """

    REQUIRED_COLUMNS = {"fecha", "idTerminal", "oper"}

    def __init__(self, cfg: dict):
        self.path       = cfg["paths"]["raw_data"]
        self.sep        = cfg["data"]["sep"]
        self.date_col   = cfg["data"]["date_col"]
        self.term_col   = cfg["data"]["terminal_col"]
        self.oper_col   = cfg["data"]["oper_col"]

    def run(self) -> pd.DataFrame:
        """
        Ejecuta la ingesta completa.

        Retorna
        ───────
        pd.DataFrame con columnas: fecha (datetime), idTerminal (int), oper (int),
        fecha_dia (date), hora (int), dow (int).
        """
        logger.info("Iniciando ingesta desde: %s", self.path)
        df = self._load()
        df = self._validate(df)
        df = self._clean(df)
        df = self._enrich(df)
        logger.info(
            "Ingesta completa — %d registros | %d terminales | %d operaciones | %s → %s",
            len(df),
            df[self.term_col].nunique(),
            df[self.oper_col].nunique(),
            df["fecha_dia"].min(),
            df["fecha_dia"].max(),
        )
        return df

    # ── Pasos internos ────────────────────────────────────────────────────

    def _load(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(
                self.path,
                sep=self.sep,
                parse_dates=[self.date_col],
            )
            logger.info("Archivo cargado: %d filas, %d columnas", *df.shape)
            return df
        except FileNotFoundError:
            logger.error("Archivo no encontrado: %s", self.path)
            raise
        except Exception as e:
            logger.error("Error al cargar el archivo: %s", e)
            raise

    def _validate(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Columnas faltantes en el dataset: {missing}")

        nulls = df[list(self.REQUIRED_COLUMNS)].isnull().sum()
        if nulls.any():
            logger.warning("Valores nulos detectados:\n%s", nulls[nulls > 0])
        else:
            logger.info("Sin valores nulos en columnas requeridas.")
        return df

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        n_before = len(df)
        df = df.drop_duplicates()
        n_dup = n_before - len(df)
        if n_dup:
            logger.warning("Eliminados %d registros duplicados.", n_dup)

        df[self.term_col] = df[self.term_col].astype(int)
        df[self.oper_col] = df[self.oper_col].astype(int)
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        return df

    def _enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """Añade columnas derivadas de la fecha para uso en EDA y modelos."""
        df["fecha_dia"] = df[self.date_col].dt.normalize()
        df["hora"]      = df[self.date_col].dt.hour
        df["dow"]       = df[self.date_col].dt.dayofweek   
        df["semana"]    = df[self.date_col].dt.to_period("W").apply(
            lambda r: r.start_time
        )
        return df