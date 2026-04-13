"""
models/holtwinters_model.py
───────────────────────────
Implementación de Holt-Winters aditivo estacional (ETS(A,A,A))
heredando de BaseModel.
"""

import logging
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from .base_model import BaseModel

logger = logging.getLogger(__name__)


class HoltWintersModel(BaseModel):

    @property
    def name(self) -> str:
        return "HoltWinters"

    def fit(self, y_train: np.ndarray) -> "HoltWintersModel":
        cfg = self.cfg
        # Reemplazar ceros para evitar problemas numéricos en ETS aditivo
        y_adj = np.where(
            y_train == 0,
            cfg["zero_replacement"],
            y_train.astype(float),
        )
        try:
            self._model = ExponentialSmoothing(
                y_adj,
                trend=cfg["trend"],
                seasonal=cfg["seasonal"],
                seasonal_periods=cfg["seasonal_periods"],
                initialization_method=cfg["initialization_method"],
            ).fit(optimized=cfg["optimized"])
            self.aic       = self._model.aic
            self.is_fitted = True
            logger.debug("HoltWinters ajustado — AIC=%.2f", self.aic)
        except Exception as e:
            logger.warning("HoltWinters falló en fit(): %s. Usando fallback media móvil.", e)
            self._fallback_mean = float(np.mean(y_train[-7:]))
            self._model         = None
            self.is_fitted      = True   # marcado como ajustado para continuar el pipeline
        return self

    def predict(self, n_periods: int) -> np.ndarray:
        self._assert_fitted()
        if self._model is None:
            return self._clip(np.full(n_periods, self._fallback_mean))
        raw = self._model.forecast(n_periods)
        return self._clip(raw)

    def forecast_with_ic(
        self,
        n_periods: int,
        q_low: np.ndarray,
        q_high: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._assert_fitted()
        pred  = self.predict(n_periods).astype(float)
        lower = np.maximum(pred + q_low[:n_periods],  0).round().astype(int)
        upper = (pred + q_high[:n_periods]).round().astype(int)
        return pred.astype(int), lower, upper

    @property
    def resid(self) -> np.ndarray:
        """Residuos del ajuste en entrenamiento (para diagnósticos)."""
        self._assert_fitted()
        if self._model is None:
            return np.array([0.0])
        return self._model.resid