"""
models/prophet_model.py
───────────────────────
Implementación de Prophet 
heredando de BaseModel.
"""

import logging
import numpy as np
import pandas as pd
from prophet import Prophet as _Prophet
from .base_model import BaseModel

logger = logging.getLogger(__name__)


class ProphetModel(BaseModel):

    @property
    def name(self) -> str:
        return "Prophet"

    def fit(self, y_train: np.ndarray) -> "ProphetModel":
        cfg = self.cfg
        # Prophet requiere DataFrame con columnas ds, y
        n = len(y_train)
        # Generamos fechas sintéticas diarias; el modelo solo necesita
        # la estructura temporal relativa, no las fechas absolutas aquí.
        # Las fechas reales se pasan en predict() via make_future_dataframe.
        self._train_len = n
        self._y_train   = y_train.astype(float)
        try:
            self._model = _Prophet(
                yearly_seasonality=cfg["yearly_seasonality"],
                weekly_seasonality=cfg["weekly_seasonality"],
                daily_seasonality=cfg["daily_seasonality"],
                seasonality_mode=cfg["seasonality_mode"],
                changepoint_prior_scale=cfg["changepoint_prior_scale"],
                interval_width=cfg["interval_width"],
            )
            # Construimos ds como fechas desde un origen arbitrario
            origin = pd.Timestamp("2017-01-31")
            ds = pd.date_range(origin, periods=n, freq="D")
            train_df = pd.DataFrame({"ds": ds, "y": y_train.astype(float)})
            self._origin = origin
            self._model.fit(train_df, algorithm=cfg["algorithm"])
            self.is_fitted = True
            logger.debug("Prophet ajustado sobre %d observaciones.", n)
        except Exception as e:
            logger.warning("Prophet falló en fit(): %s. Usando fallback.", e)
            self._fallback_mean = float(np.mean(y_train[-7:]))
            self._model         = None
            self.is_fitted      = True
        return self

    def predict(self, n_periods: int) -> np.ndarray:
        self._assert_fitted()
        if self._model is None:
            return self._clip(np.full(n_periods, self._fallback_mean))
        future = self._model.make_future_dataframe(
            periods=n_periods, freq="D"
        )
        fc  = self._model.predict(future)
        raw = fc["yhat"].values[-n_periods:]
        return self._clip(raw)

    def forecast_with_ic(
        self,
        n_periods: int,
        q_low: np.ndarray,
        q_high: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._assert_fitted()
        if self._model is None:
            pred  = self._clip(np.full(n_periods, self._fallback_mean))
            return pred, np.maximum(pred + q_low[:n_periods], 0).astype(int), (pred + q_high[:n_periods]).astype(int)
        future = self._model.make_future_dataframe(periods=n_periods, freq="D")
        fc     = self._model.predict(future)
        pred   = self._clip(fc["yhat"].values[-n_periods:]).astype(float)
        lower  = np.maximum(pred + q_low[:n_periods],  0).round().astype(int)
        upper  = (pred + q_high[:n_periods]).round().astype(int)
        return pred.astype(int), lower, upper