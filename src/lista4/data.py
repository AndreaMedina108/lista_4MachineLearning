import pandas as pd

def cargar_datos(filepath):
    """Carga un archivo CSV y retorna un DataFrame de pandas."""
    return pd.read_csv(filepath)