import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



FIGURAS_DIR = "figuras"
os.makedirs(FIGURAS_DIR, exist_ok=True)

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]


# -----------------------------------------------------------------------------
# Univariada
# -----------------------------------------------------------------------------

def analise_univariada(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula estatísticas descritivas e gera histogramas + boxplots.

    Returns:
        DataFrame com as estatísticas resumidas
    """
    print("\n  → Estatísticas descritivas:")
    stats_df = df[FEATURES].describe().T
    stats_df["skewness"] = df[FEATURES].skew()
    stats_df["kurtosis"] = df[FEATURES].kurt()
    print(stats_df.round(2).to_string())

    # Histogramas
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i, col in enumerate(FEATURES):
        axes[i].hist(df[col], bins=30, color="#4C8BE2", edgecolor="white", alpha=0.85)
        axes[i].set_title(col, fontsize=11, fontweight="bold")
        axes[i].set_xlabel("Valor")
        axes[i].set_ylabel("Frequência")
    fig.suptitle("Distribuição das Variáveis (Histogramas)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{FIGURAS_DIR}/01_histogramas.png", dpi=150)
    plt.close()
    print(f"  → Salvo: {FIGURAS_DIR}/01_histogramas.png")

    # Boxplots
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i, col in enumerate(FEATURES):
        axes[i].boxplot(df[col], patch_artist=True,
                        boxprops=dict(facecolor="#4C8BE2", color="#1a1a2e"),
                        medianprops=dict(color="white", linewidth=2))
        axes[i].set_title(col, fontsize=11, fontweight="bold")
        axes[i].set_ylabel("Valor")
    fig.suptitle("Distribuição das Variáveis (Boxplots)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{FIGURAS_DIR}/02_boxplots.png", dpi=150)
    plt.close()
    print(f"  → Salvo: {FIGURAS_DIR}/02_boxplots.png")

    return stats_df


# -----------------------------------------------------------------------------
# Bivariada
# -----------------------------------------------------------------------------

def analise_bivariada(df: pd.DataFrame) -> None:
    """
    Gera matriz de dispersão e calcula correlações de Pearson e Spearman.
    """
    # Pearson
    pearson = df[FEATURES].corr(method="pearson")
    print("\n  → Correlação de Pearson (top pares):")
    pares = (
        pearson.where(np.triu(np.ones(pearson.shape), k=1).astype(bool))
        .stack()
        .reset_index()
    )
    pares.columns = ["Var1", "Var2", "Pearson"]
    pares = pares.reindex(pares["Pearson"].abs().sort_values(ascending=False).index)
    print(pares.head(10).to_string(index=False))

    # Spearman
    spearman = df[FEATURES].corr(method="spearman")

    # Scatter matrix (amostra para não ficar lento)
    sample = df[FEATURES].sample(min(300, len(df)), random_state=42)
    fig = sns.pairplot(sample, diag_kind="kde", plot_kws={"alpha": 0.4, "s": 15, "color": "#4C8BE2"})
    fig.figure.suptitle("Matriz de Dispersão (amostra 300)", y=1.02, fontsize=13, fontweight="bold")
    plt.savefig(f"{FIGURAS_DIR}/03_scatter_matrix.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  → Salvo: {FIGURAS_DIR}/03_scatter_matrix.png")


# -----------------------------------------------------------------------------
# Multivariada
# -----------------------------------------------------------------------------

def analise_multivariada(df: pd.DataFrame) -> None:
    """
    Gera heatmap de correlação e explica a variância por PCA.
    """
    
    # Heatmap de correlação
    corr = df[FEATURES].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        vmin=-1, vmax=1, linewidths=0.5, ax=ax,
        annot_kws={"size": 9}
    )
    ax.set_title("Heatmap de Correlação de Pearson", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{FIGURAS_DIR}/04_heatmap_correlacao.png", dpi=150)
    plt.close()
    print(f"  → Salvo: {FIGURAS_DIR}/04_heatmap_correlacao.png")

    # PCA - variância explicada
    X = StandardScaler().fit_transform(df[FEATURES])
    pca = PCA()
    pca.fit(X)
    variancia_acumulada = np.cumsum(pca.explained_variance_ratio_) * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(1, len(FEATURES) + 1), pca.explained_variance_ratio_ * 100,
           color="#4C8BE2", edgecolor="white", label="Variância individual")
    ax.plot(range(1, len(FEATURES) + 1), variancia_acumulada,
            color="#E24C4C", marker="o", linewidth=2, label="Variância acumulada")
    ax.axhline(80, color="gray", linestyle="--", linewidth=1, label="80% threshold")
    ax.set_xlabel("Componente Principal")
    ax.set_ylabel("Variância Explicada (%)")
    ax.set_title("Análise PCA - Variância Explicada por Componente", fontsize=13, fontweight="bold")
    ax.legend()
    ax.set_xticks(range(1, len(FEATURES) + 1))
    plt.tight_layout()
    plt.savefig(f"{FIGURAS_DIR}/05_pca_variancia.png", dpi=150)
    plt.close()
    print(f"  → Salvo: {FIGURAS_DIR}/05_pca_variancia.png")

    print("\n  → Variância acumulada por componente PCA:")
    for i, v in enumerate(variancia_acumulada, 1):
        print(f"    PC{i}: {v:.1f}%")