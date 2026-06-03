import numpy as np
import pandas as pd 
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

def calcular_metricas(X: np.ndarray, labels: np.ndarray) -> dict:
    mascara = labels != -1
    X_valido = X[mascara]
    labels_validos = labels[mascara]

    n_clusters = len(set(labels_validos))

    if n_clusters < 2:
        return {
            "silhouette": None,
            "davies_bouldin": None,
            "calinski_harabasz": None,
        }
    
    return {
        "silhouette":        round(silhouette_score(X_valido, labels_validos), 4),
        "davies_bouldin":    round(davies_bouldin_score(X_valido, labels_validos), 4),
        "calinski_harabasz": round(calinski_harabasz_score(X_valido, labels_validos), 2),
    }

def avaliar_todos(resultado: list[dict], X: np.ndarray) -> list[dict]:
    for r in resultado: 
        r["metricas"] = calcular_metricas(X, r["labels"])
    return resultado

def exibir_tabela_resultados(resultados: list[dict]) -> pd.DataFrame:
    linhas = []
    for r in resultados:
        metricas = r["metricas"]
        linhas.append({
            "Algoritmo":           r["algoritmo"],
            "Hiperparâmetros":     r["hiperparametros"],
            "Silhouette ↑":        metricas["silhouette"],
            "Davies-Bouldin ↓":    metricas["davies_bouldin"],
            "Calinski-Harabasz ↑": metricas["calinski_harabasz"],

        })

        df = pd.DataFrame(linhas)
        df_valido = df.dropna()

    print("\n" + "=" * 85)
    print("  TABELA DE RESULTADOS")
    print("=" * 85)
    print(df.to_string(index=False))
