import numpy as np

def riesgo_empirico_cuadratico(y_true, y_pred):
    """Calcula el riesgo empírico para pérdida cuadrática."""
    return np.mean((y_true - y_pred) ** 2)

def riesgo_empirico_logistico(y_true, y_prob):
    """Calcula el riesgo empírico logístico."""
    # Se agrega un epsilon minúsculo para prevenir errores matemáticos de log(0)
    eps = 1e-15
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))

def errores_clasificacion(y_true, y_pred_class):
    """Retorna número de errores y su proporción."""
    errores = np.sum(y_true != y_pred_class)
    proporcion = errores / len(y_true)
    return errores, proporcion