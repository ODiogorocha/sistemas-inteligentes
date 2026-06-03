import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

FIGURAS_DIR = "figuras"
os.makedirs(FIGURAS_DIR, exist_ok=True)

PALETA = ["#4C8BE2", "#E24C4C", "#4CE27A", "#E2C84C", "#A04CE2", "#4CE2E2"]


def _cor_para_label(labels: np.ndarray) -> list[str]:
    """Mapeia labels inteiros para cores da paleta (ruído DBSCAN = cinza)."""
    cores = []
    for lbl in labels:
        if lbl == -1:
            cores.append("#AAAAAA")
        else:
            cores.append(PALETA[lbl % len(PALETA)])
    return cores


def plotar_clusters_pca(
    resultados: list[dict],
    X_pca: np.ndarray,
    rotulos_originais: np.ndarray,
) -> None:
    """
    Gera um subplot 2D para cada configuração de clustering testada,
    colorindo os pontos pelos clusters encontrados. Também salva um
    gráfico separado com os rótulos originais (Outcome).

    Args:
        resultados         : lista de dicts com labels e métricas
        X_pca              : array (N, 2) com componentes principais
        rotulos_originais  : array com Outcome (0/1)
    """
    # --- Gráfico: rótulos originais ---
    fig, ax = plt.subplots(figsize=(7, 5))
    cores_orig = [PALETA[0] if r == 0 else PALETA[1] for r in rotulos_originais]
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=cores_orig, alpha=0.6, s=18, edgecolors="none")
    legenda = [
        mpatches.Patch(color=PALETA[0], label="Sem Diabetes (0)"),
        mpatches.Patch(color=PALETA[1], label="Com Diabetes (1)"),
    ]
    ax.legend(handles=legenda, fontsize=9)
    ax.set_title("Rótulos Originais no Espaço PCA", fontsize=12, fontweight="bold")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.tight_layout()
    plt.savefig(f"{FIGURAS_DIR}/06_rotulos_originais_pca.png", dpi=150)
    plt.close()
    print(f"  → Salvo: {FIGURAS_DIR}/06_rotulos_originais_pca.png")

    # --- Subplots por algoritmo ---
    _plotar_grupo(resultados, X_pca, "K-Means",      "07_kmeans_clusters.png")
    _plotar_grupo(resultados, X_pca, "Agglomerative", "08_agglomerative_clusters.png")
    _plotar_grupo(resultados, X_pca, "DBSCAN",        "09_dbscan_clusters.png")


def _plotar_grupo(
    resultados: list[dict],
    X_pca: np.ndarray,
    algoritmo: str,
    nome_arquivo: str,
) -> None:
    """Gera figura com subplots para todas as configs de um algoritmo."""
    grupo = [r for r in resultados if r["algoritmo"] == algoritmo]
    if not grupo:
        return

    n = len(grupo)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = np.array(axes).flatten()

    for i, r in enumerate(grupo):
        ax = axes[i]
        labels = r["labels"]
        cores = _cor_para_label(labels)
        ax.scatter(X_pca[:, 0], X_pca[:, 1], c=cores, alpha=0.65, s=15, edgecolors="none")

        # Legenda de clusters
        clusters_unicos = sorted(set(labels))
        patches = []
        for c in clusters_unicos:
            cor = "#AAAAAA" if c == -1 else PALETA[c % len(PALETA)]
            label_txt = f"Ruído" if c == -1 else f"Cluster {c}"
            patches.append(mpatches.Patch(color=cor, label=label_txt))
        ax.legend(handles=patches, fontsize=7, loc="upper right")

        # Métricas no título
        m = r["metricas"]
        if m["silhouette"] is not None:
            titulo_metricas = (
                f"Sil={m['silhouette']:.3f}  "
                f"DB={m['davies_bouldin']:.3f}  "
                f"CH={m['calinski_harabasz']:.0f}"
            )
        else:
            titulo_metricas = "clusters insuficientes para métricas"

        ax.set_title(
            f"{r['hiperparametros']}\n{titulo_metricas}",
            fontsize=8, fontweight="bold"
        )
        ax.set_xlabel("PC1", fontsize=8)
        ax.set_ylabel("PC2", fontsize=8)

    # Esconder eixos extras
    for j in range(len(grupo), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Clustering: {algoritmo}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{FIGURAS_DIR}/{nome_arquivo}", dpi=150)
    plt.close()
    print(f"  → Salvo: {FIGURAS_DIR}/{nome_arquivo}")