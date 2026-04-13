"""
models/base_model.py
────────────────────
Define la interfaz común que todos los modelos deben cumplir.

Cualquier modelo nuevo que se quiera agregar al pipeline solo necesita:
  1. Heredar de BaseModel.
  2. Implementar los tres métodos abstractos: fit, predict, forecast.

"""

from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    """
    Interfaz abstracta para modelos de series de tiempo.

    Atributos esperados después de fit()
    ─────────────────────────────────────
    name      : str   — nombre legible del modelo
    is_fitted : bool  — True si fit() fue llamado exitosamente
    aic       : float — criterio de información (None si no aplica)
    """

    def __init__(self, cfg: dict):
        """
        cfg : sub-diccionario del config.yaml específico para este modelo.
        """
        self.cfg       = cfg
        self.is_fitted = False
        self.aic       = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Nombre identificador del modelo (ej. 'HoltWinters')."""
        ...

    @abstractmethod
    def fit(self, y_train: np.ndarray) -> "BaseModel":
        """
        Ajusta el modelo sobre la serie de entrenamiento.

        Parámetros
        ──────────
        y_train : array 1-D de floats/ints — serie temporal de entrenamiento

        Retorna
        ───────
        self  (permite encadenamiento: model.fit(y).predict(n))
        """
        ...

    @abstractmethod
    def predict(self, n_periods: int) -> np.ndarray:
        """
        Genera predicciones puntuales para los próximos n_periods pasos.

        Debe llamarse después de fit(). Retorna valores >= 0 (clipeados).

        Retorna
        ───────
        np.ndarray de shape (n_periods,) con valores enteros >= 0
        """
        ...

    @abstractmethod
    def forecast_with_ic(
        self,
        n_periods: int,
        q_low: np.ndarray,
        q_high: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Genera predicción + IC empírico.

        Parámetros
        ──────────
        n_periods : horizonte de predicción
        q_low     : array (n_periods,) — percentil inferior de errores (rolling CV)
        q_high    : array (n_periods,) — percentil superior de errores (rolling CV)

        Retorna
        ───────
        (pred, lower, upper) — tres arrays de shape (n_periods,)
        """
        ...

    # ── Métodos concretos compartidos ─────────────────────────────────────

    def _assert_fitted(self):
        if not self.is_fitted:
            raise RuntimeError(
                f"El modelo '{self.name}' debe ser ajustado con fit() antes de predecir."
            )

    @staticmethod
    def _clip(arr: np.ndarray) -> np.ndarray:
        """Clipea a 0 y redondea a entero."""
        return np.clip(arr, 0, None).round().astype(int)