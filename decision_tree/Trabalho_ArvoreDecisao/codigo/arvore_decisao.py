"""
Trabalho Prático: Construção e Análise de Árvores de Decisão
Dataset: Heart Disease (Doença Cardíaca)
Biblioteca: Scikit-Learn
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Resolve paths relativos à localização deste script,
# independente de onde o script for executado
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET     = os.path.join(BASE_DIR, "..", "dataset", "heart_disease.csv")
GRAFICOS    = os.path.join(BASE_DIR, "..", "graficos")

os.makedirs(GRAFICOS, exist_ok=True)

# =============================================================================
# 1. CARREGAMENTO E PRÉ-PROCESSAMENTO DOS DADOS
# =============================================================================

print("=" * 60)
print("1. CARREGAMENTO E PRÉ-PROCESSAMENTO DOS DADOS")
print("=" * 60)

df = pd.read_csv(DATASET)

print(f"\nShape do dataset: {df.shape}")
print(f"\nPrimeiras linhas:\n{df.head()}")
print(f"\nInformações gerais:")
df.info()
print(f"\nValores nulos:\n{df.isnull().sum()}")
print(f"\nEstatísticas descritivas:\n{df.describe()}")

X = df.drop("target", axis=1)
y = df["target"]

print(f"\nDistribuição da variável alvo:\n{y.value_counts()}")
print(f"Proporção: {y.value_counts(normalize=True).round(2).to_dict()}")

# Divisão treino/teste 70/30, estratificado
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"\nTamanho do treino: {X_train.shape[0]} amostras")
print(f"Tamanho do teste:  {X_test.shape[0]} amostras")

# =============================================================================
# 2. FUNÇÃO AUXILIAR
# =============================================================================

def avaliar(max_depth, criterion, min_samples_leaf, min_samples_split):
    m = DecisionTreeClassifier(
        max_depth=max_depth,
        criterion=criterion,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        random_state=42
    )
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)
    return (
        round(accuracy_score(y_test, y_pred), 4),
        round(precision_score(y_test, y_pred, zero_division=0), 4),
        round(recall_score(y_test, y_pred, zero_division=0), 4),
    )

# =============================================================================
# 3. EXPERIMENTOS
# =============================================================================

print("\n" + "=" * 60)
print("2. EXPERIMENTOS COM DIFERENTES CONFIGURAÇÕES")
print("=" * 60)

configs = [
    {"max_depth": 3,    "criterion": "gini",    "min_samples_leaf": 1,  "min_samples_split": 2},
    {"max_depth": 5,    "criterion": "gini",    "min_samples_leaf": 1,  "min_samples_split": 2},
    {"max_depth": 10,   "criterion": "gini",    "min_samples_leaf": 1,  "min_samples_split": 2},
    {"max_depth": None, "criterion": "gini",    "min_samples_leaf": 1,  "min_samples_split": 2},
    {"max_depth": 5,    "criterion": "entropy", "min_samples_leaf": 1,  "min_samples_split": 2},
    {"max_depth": 5,    "criterion": "gini",    "min_samples_leaf": 5,  "min_samples_split": 2},
    {"max_depth": 5,    "criterion": "gini",    "min_samples_leaf": 10, "min_samples_split": 2},
    {"max_depth": 5,    "criterion": "gini",    "min_samples_leaf": 1,  "min_samples_split": 5},
    {"max_depth": 5,    "criterion": "gini",    "min_samples_leaf": 1,  "min_samples_split": 10},
    {"max_depth": 3,    "criterion": "entropy", "min_samples_leaf": 1,  "min_samples_split": 2},
    {"max_depth": 5,    "criterion": "entropy", "min_samples_leaf": 5,  "min_samples_split": 5},
    {"max_depth": 3,    "criterion": "entropy", "min_samples_leaf": 5,  "min_samples_split": 5},
]

results = []
for i, cfg in enumerate(configs):
    acc, prec, rec = avaliar(
        cfg["max_depth"], cfg["criterion"],
        cfg["min_samples_leaf"], cfg["min_samples_split"]
    )
    results.append({
        "Modelo": f"M{i+1:02d}",
        "max_depth": str(cfg["max_depth"]),
        "criterion": cfg["criterion"],
        "min_samples_leaf": cfg["min_samples_leaf"],
        "min_samples_split": cfg["min_samples_split"],
        "Acuracia": acc, "Precisao": prec, "Recall": rec,
    })
    print(f"\nModelo {i+1:02d}: depth={cfg['max_depth']}, criterion={cfg['criterion']}, "
          f"leaf={cfg['min_samples_leaf']}, split={cfg['min_samples_split']}")
    print(f"  Acurácia: {acc:.4f} | Precisão: {prec:.4f} | Recall: {rec:.4f}")

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("3. TABELA COMPARATIVA")
print("=" * 60)
print(results_df.to_string(index=False))

best_idx = results_df["Acuracia"].idxmax()
best = results_df.loc[best_idx]
print(f"\nMelhor modelo: {best['Modelo']} | depth={best['max_depth']} | "
      f"criterion={best['criterion']} | leaf={best['min_samples_leaf']} | "
      f"split={best['min_samples_split']}")
print(f"Acurácia: {best['Acuracia']} | Precisão: {best['Precisao']} | Recall: {best['Recall']}")

# =============================================================================
# 4. GRÁFICOS
# =============================================================================

BLUE = "#2196F3"; GREEN = "#4CAF50"; RED = "#e53935"; ORANGE = "#FF9800"

# --- Figura 1: Visão geral ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Figura 1 – Comparação Geral dos Modelos", fontsize=13, fontweight="bold")
x = np.arange(len(results_df))
for ax, metric, label, color in zip(axes,
        ["Acuracia","Precisao","Recall"],
        ["Acurácia","Precisão","Recall"],
        [BLUE, GREEN, ORANGE]):
    vals = results_df[metric]
    bar_colors = [GREEN if i == best_idx else color for i in range(len(vals))]
    bars = ax.bar(x, vals, color=bar_colors, alpha=0.85, width=0.6)
    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.set_xlabel("Modelos"); ax.set_ylabel("Score")
    ax.set_xticks(x); ax.set_xticklabels(results_df["Modelo"], rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0.6, 1.05)
    ax.axhline(vals.mean(), color=RED, linestyle="--", linewidth=1, label=f"Média: {vals.mean():.2f}")
    ax.legend(fontsize=8)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f"{val:.2f}", ha="center", va="bottom", fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(GRAFICOS, "fig1_visao_geral.png"), dpi=150, bbox_inches="tight")
plt.close()
print("\nGráfico salvo: fig1_visao_geral.png")

# --- Figura 2: Efeito max_depth ---
depths = [3, 5, 10, None]
dlabels = ["3", "5", "10", "Ilimitado"]
res_d = [avaliar(d, "gini", 1, 2) for d in depths]
acc_d, prec_d, rec_d = zip(*res_d)
fig2, ax2 = plt.subplots(figsize=(7, 5))
ax2.plot(dlabels, acc_d,  "o-",  color=BLUE,   label="Acurácia",  linewidth=2)
ax2.plot(dlabels, prec_d, "s--", color=GREEN,  label="Precisão",  linewidth=2)
ax2.plot(dlabels, rec_d,  "^-.", color=ORANGE, label="Recall",    linewidth=2)
ax2.set_title("Figura 2 – Efeito do max_depth\n(criterion=gini, leaf=1, split=2)",
              fontsize=11, fontweight="bold")
ax2.set_xlabel("max_depth"); ax2.set_ylabel("Score")
ax2.set_ylim(0.5, 1.05); ax2.legend(); ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(GRAFICOS, "fig2_max_depth.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Gráfico salvo: fig2_max_depth.png")

# --- Figura 3: Efeito criterion ---
criterions = ["gini", "entropy"]
res_c = [avaliar(5, c, 1, 2) for c in criterions]
acc_c, prec_c, rec_c = zip(*res_c)
x_c = np.arange(len(criterions)); w = 0.25
fig3, ax3 = plt.subplots(figsize=(6, 5))
b1 = ax3.bar(x_c-w, acc_c,  w, label="Acurácia", color=BLUE,   alpha=0.85)
b2 = ax3.bar(x_c,   prec_c, w, label="Precisão", color=GREEN,  alpha=0.85)
b3 = ax3.bar(x_c+w, rec_c,  w, label="Recall",   color=ORANGE, alpha=0.85)
ax3.set_title("Figura 3 – Efeito do criterion\n(max_depth=5, leaf=1, split=2)",
              fontsize=11, fontweight="bold")
ax3.set_xlabel("criterion"); ax3.set_ylabel("Score")
ax3.set_xticks(x_c); ax3.set_xticklabels(criterions, fontsize=10)
ax3.set_ylim(0.5, 1.05); ax3.legend(); ax3.grid(True, alpha=0.3, axis="y")
for bars, vals in [(b1,acc_c),(b2,prec_c),(b3,rec_c)]:
    for bar, val in zip(bars, vals):
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(GRAFICOS, "fig3_criterion.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Gráfico salvo: fig3_criterion.png")

# --- Figura 4: Efeito min_samples_leaf ---
leaves = [1, 2, 5, 10, 15, 20]
res_l = [avaliar(5, "gini", l, 2) for l in leaves]
acc_l, prec_l, rec_l = zip(*res_l)
fig4, ax4 = plt.subplots(figsize=(7, 5))
ax4.plot([str(l) for l in leaves], acc_l,  "o-",  color=BLUE,   label="Acurácia",  linewidth=2)
ax4.plot([str(l) for l in leaves], prec_l, "s--", color=GREEN,  label="Precisão",  linewidth=2)
ax4.plot([str(l) for l in leaves], rec_l,  "^-.", color=ORANGE, label="Recall",    linewidth=2)
ax4.set_title("Figura 4 – Efeito do min_samples_leaf\n(max_depth=5, criterion=gini, split=2)",
              fontsize=11, fontweight="bold")
ax4.set_xlabel("min_samples_leaf"); ax4.set_ylabel("Score")
ax4.set_ylim(0.5, 1.05); ax4.legend(); ax4.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(GRAFICOS, "fig4_min_samples_leaf.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Gráfico salvo: fig4_min_samples_leaf.png")

# --- Figura 5: Efeito min_samples_split ---
splits = [2, 5, 10, 15, 20]
res_s = [avaliar(5, "gini", 1, s) for s in splits]
acc_s, prec_s, rec_s = zip(*res_s)
fig5, ax5 = plt.subplots(figsize=(7, 5))
ax5.plot([str(s) for s in splits], acc_s,  "o-",  color=BLUE,   label="Acurácia",  linewidth=2)
ax5.plot([str(s) for s in splits], prec_s, "s--", color=GREEN,  label="Precisão",  linewidth=2)
ax5.plot([str(s) for s in splits], rec_s,  "^-.", color=ORANGE, label="Recall",    linewidth=2)
ax5.set_title("Figura 5 – Efeito do min_samples_split\n(max_depth=5, criterion=gini, leaf=1)",
              fontsize=11, fontweight="bold")
ax5.set_xlabel("min_samples_split"); ax5.set_ylabel("Score")
ax5.set_ylim(0.5, 1.05); ax5.legend(); ax5.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(GRAFICOS, "fig5_min_samples_split.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Gráfico salvo: fig5_min_samples_split.png")

print("\n" + "=" * 60)
print("Script executado com sucesso!")
print("=" * 60)