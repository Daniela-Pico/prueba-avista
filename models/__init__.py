from .holtwinters_model import HoltWintersModel
from .prophet_model     import ProphetModel
from .sarima_model      import SarimaModel
from .base_model        import BaseModel

MODEL_REGISTRY = {
    "HoltWinters": HoltWintersModel,
    "Prophet"    : ProphetModel,
    "SARIMA"     : SarimaModel,
}