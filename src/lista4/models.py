from sklearn.linear_model import LinearRegression, LogisticRegression

def ajustar_regresion_lineal(X, y):
    """Ajusta un modelo de regresión lineal con intercepto."""
    modelo = LinearRegression(fit_intercept=True)
    modelo.fit(X, y)
    return modelo

def ajustar_regresion_logistica(X, y):
    """Ajusta regresión logística con solver lbfgs y C=1e6."""
    modelo = LogisticRegression(solver='lbfgs', C=1e6, max_iter=2000)
    modelo.fit(X, y)
    return modelo