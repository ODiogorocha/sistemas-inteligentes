import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

def rodar_kmeans(X: np.ndarray) -> list[dict]:
    resultados = []
    ks = [2,3,4]

    print("  → K-Means...")

    for k in ks:
        modelo = KMeans(n_clusters=k, init="k-means++", n_init=20, random_state=42)
        labels = modelo.fit_predict(X)
        n_clusters_efetivos = len(set(labels))
        print(f"    k={k} → {n_clusters_efetivos} clusters encontrados")
        resultados.append({
            "algoritmo": "K-Means",
            "hiperparametros": f"k={k}",
            "labels": labels,
            "modelo": modelo,
        })

    return resultados

def rodar_agglomerative(X: np.ndarray) -> list[dict]:
    resultados = []
    configs = [
        {"n_clusters": 2, "linkage": "ward"},
        {"n_clusters": 3, "linkage": "ward"},
        {"n_clusters": 2, "linkage": "complete"},
        {"n_clusters": 3, "linkage": "complete"},
        {"n_clusters": 2, "linkage": "average"},
        {"n_clusters": 3, "linkage": "average"},
    ]

    print("  → Agglomerative Clustering...")
    for cfg in configs:
        modelo = AgglomerativeClustering(
            n_clusters=cfg["n_clusters"],
            linkage=cfg["linkage"]
        )
        labels = modelo.fit_predict(X)
        desc = f"k={cfg['n_clusters']}, linkage={cfg['linkage']}"
        print(f"    {desc} → {cfg['n_clusters']} clusters")
        resultados.append({
            "algoritmo": "Agglomerative",
            "hiperparametros": desc,
            "labels": labels,
            "modelo": modelo,
        })

    return resultados

def rodar_dbscan(X: np.ndarray) -> list[dict]:
    resultados = []
    configs = [
        {"eps": 0.5, "min_samples": 5},
        {"eps": 0.7, "min_samples": 5},
        {"eps": 0.5, "min_samples": 10},
        {"eps": 1.0, "min_samples": 5},
        {"eps": 1.0, "min_samples": 10},
    ]

    print("  → DBSCAN...")
    for cfg in configs:
        modelo = DBSCAN(eps=cfg["eps"], min_samples=cfg["min_samples"])
        labels = modelo.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_ruido = (labels == -1).sum()
        desc = f"eps={cfg['eps']}, min_samples={cfg['min_samples']}"
        print(f"    {desc} → {n_clusters} clusters, {n_ruido} pontos de ruído")
        resultados.append({
            "algoritmo": "DBSCAN",
            "hiperparametros": desc,
            "labels": labels,
            "modelo": modelo,
            "n_ruido": n_ruido,
        })

    return resultados
