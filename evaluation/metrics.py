
import logging
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)


# ── Métricas de error ────────────────────────────────────────────────────────

def compute_metrics(
    y_true     : np.ndarray,
    y_pred     : np.ndarray,
    model_name : str = "",
    terminal   : int = None,
    oper       : int = None,
) -> dict:
    """
    Calcula MAE, RMSE y MAPE para un par (real, predicho).

    MAPE: se excluyen los días con y_true == 0 para evitar
    divisiones por cero.
    """
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    denom = np.where(y_true == 0, np.nan, y_true)
    mape  = np.nanmean(np.abs((y_true - y_pred) / denom)) * 100

    return {
        "idTerminal": terminal,
        "oper"      : oper,
        "modelo"    : model_name,
        "MAE"       : round(mae,  2),
        "RMSE"      : round(rmse, 2),
        "MAPE_%"    : round(mape, 1),
        "n_test"    : len(y_true),
    }


def empirical_ic(
    y_train   : np.ndarray,
    n_test    : int,
    model_cls,
    model_cfg : dict,
    alpha     : float = 0.05,
    min_factor: int   = 3,
    max_folds : int   = None,
) -> tuple:
    """
    Calcula el IC empirico al nivel (1-alpha)*100% via rolling origin CV.

    Procedimiento
    Para k = T_min ... T - n_test (maximo max_folds iteraciones):
      1. Entrenar el modelo con y_train[:k].
      2. Predecir n_test pasos adelante.
      3. Registrar error(k, h) = y_real(k+h) - y_pred(k+h) para h=1..n_test.
    Para cada horizonte h, los cuantiles alpha/2 y 1-alpha/2
    de los errores acumulados forman el IC.

    Si se limita max_folds, se toman los ULTIMOS max_folds origenes
    (ventanas con mas datos, mas representativas del estado reciente).

    Retorna (q_low, q_high, n_folds)
    """
    T     = len(y_train)
    T_min = min_factor * 7

    all_origins = list(range(T_min, T - n_test + 1))

    if max_folds is not None and len(all_origins) > max_folds:
        all_origins = all_origins[-max_folds:]

    errors = [[] for _ in range(n_test)]

    for k in all_origins:
        y_window = y_train[:k]
        y_future = y_train[k: k + n_test]
        try:
            m    = model_cls(model_cfg).fit(y_window)
            pred = m.predict(n_test)
            for h in range(min(n_test, len(y_future))):
                errors[h].append(float(y_future[h]) - float(pred[h]))
        except Exception:
            continue

    n_folds = len(errors[0]) if errors[0] else 0

    q_low  = np.array([
        np.percentile(e, 100 * alpha / 2)       if e else -np.inf
        for e in errors
    ])
    q_high = np.array([
        np.percentile(e, 100 * (1 - alpha / 2)) if e else  np.inf
        for e in errors
    ])

    logger.debug(
        "IC empirico — %d folds | alpha=%.2f | T_min=%d",
        n_folds, alpha, T_min,
    )
    return q_low, q_high, n_folds


def empirical_coverage(
    y_true: np.ndarray,
    lower : np.ndarray,
    upper : np.ndarray,
) -> float:
    """Fraccion de puntos reales que caen dentro del IC."""
    n = min(len(y_true), len(lower), len(upper))
    return float(np.mean((y_true[:n] >= lower[:n]) & (y_true[:n] <= upper[:n]))) * 100