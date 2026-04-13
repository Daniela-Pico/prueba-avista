"""
models/sarima_model.py
──────────────────────
Implementación de SARIMA con selección automática de orden via AIC
(pmdarima.auto_arima) heredando de BaseModel.
"""

import logging
import numpy as np
import pmdarima as pm
from .base_model import BaseModel

logger = logging.getLogger(__name__)


class SarimaModel(BaseModel):

    @property
    def name(self) -> str:
        return "SARIMA"

    def fit(self, y_train: np.ndarray) -> "SarimaModel":
        cfg = self.cfg
        try:
            self._model = pm.auto_arima(
                y_train,
                m=cfg["m"],
                seasonal=cfg["seasonal"],
                d=None,          # selección automática via test ADF
                D=cfg["D"],
                max_p=cfg["max_p"],
                max_q=cfg["max_q"],
                max_P=cfg["max_P"],
                max_Q=cfg["max_Q"],
                stepwise=cfg["stepwise"],
                information_criterion=cfg["ic"],
                error_action="ignore",
                suppress_warnings=True,
                n_fits=cfg["n_fits"],
            )
            self.aic       = self._model.aic()
            self.is_fitted = True
            logger.debug(
                "SARIMA ajustado — orden=%s  AIC=%.2f",
                self._model.order, self.aic,
            )
        except Exception as e:
            logger.warning("SARIMA falló en fit(): %s. Usando fallback.", e)
            self._fallback_mean = float(np.mean(y_train[-7:]))
            self._model         = None
            self.is_fitted      = True
        return self

    def predict(self, n_periods: int) -> np.ndarray:
        self._assert_fitted()
        if self._model is None:
            return self._clip(np.full(n_periods, self._fallback_mean))
        try:
            raw, _ = self._model.predict(n_periods=n_periods, return_conf_int=True)
            return self._clip(raw)
        except Exception as e:
            logger.warning("SARIMA predict() falló: %s", e)
            return self._clip(np.full(n_periods, self._fallback_mean))

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