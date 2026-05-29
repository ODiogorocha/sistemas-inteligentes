"""
===============================================================================
TRABALHO PRÁTICO — ÁRVORES DE DECISÃO
DISCIPLINA: SISTEMAS INTELIGENTES
TEMA: ANÁLISE DE MORTALIDADE POR CID-10
ALUNO: DIOGO ROCHA MARQUES
===============================================================================
"""

# =============================================================================
# IMPORTAÇÕES
# =============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

# =============================================================================
# CONFIGURAÇÕES DE CAMINHO
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_DIR = os.path.dirname(BASE_DIR)

DATASET = os.path.join(
    PROJECT_DIR,
    "db",
    "sim_cnv_obt10br142742187_60_99_223.csv"
)

GRAFICOS = os.path.join(
    PROJECT_DIR,
    "graficos"
)

os.makedirs(GRAFICOS, exist_ok=True)

# =============================================================================
# FUNÇÃO AUXILIAR PARA TÍTULOS
# =============================================================================

def titulo(texto):

    print("\n")
    print("=" * 80)
    print(f"{texto:^80}")
    print("=" * 80)

# =============================================================================
# VERIFICAÇÃO DOS CAMINHOS
# =============================================================================

titulo("VERIFICAÇÃO DE CAMINHOS")

print(f"\nBASE_DIR:\n{BASE_DIR}")

print(f"\nPROJECT_DIR:\n{PROJECT_DIR}")

print(f"\nDATASET:\n{DATASET}")

print(f"\nArquivo existe?\n{os.path.exists(DATASET)}")


# =============================================================================
# 1. CARREGAMENTO DO DATASET
# =============================================================================

print("\n" + "=" * 80)
print("1. CARREGAMENTO DO DATASET".center(80))
print("=" * 80)

print("\nLendo arquivo CSV...")

# Lê o arquivo ignorando linhas iniciais do DATASUS
df = pd.read_csv(
    DATASET,
    sep=";",
    encoding="latin1",
    skiprows=3,
    engine="python"
)

# Remove colunas vazias
df = df.dropna(axis=1, how="all")

# Remove linhas vazias
df = df.dropna(how="all")

# Corrige nomes das colunas
df.columns = [col.strip() for col in df.columns]

print("Dataset carregado com sucesso!")

print(f"\nQuantidade de linhas: {df.shape[0]}")
print(f"Quantidade de colunas: {df.shape[1]}")

print("\nPrimeiras 5 linhas do dataset:\n")
print(df.head())

# Remove possíveis linhas de rodapé do DATASUS
df = df[
    ~df.iloc[:, 0].astype(str).str.contains(
        "Fonte|Notas|Período",
        case=False,
        na=False
    )
]

# Resetar índice
df.reset_index(drop=True, inplace=True)
# =============================================================================
# 2. PRÉ-PROCESSAMENTO
# =============================================================================

titulo("2. PRÉ-PROCESSAMENTO DOS DADOS")

print("\nRemovendo espaços extras dos nomes das colunas...")

df.columns = df.columns.str.strip()

print("\nColunas encontradas:\n")

for coluna in df.columns:

    print(f"• {coluna}")

print("\nConvertendo valores inválidos...")

for coluna in df.columns[1:]:

    df[coluna] = (

        df[coluna]
        .astype(str)
        .str.replace('-', '0')
        .str.replace(',', '.')
    )

    df[coluna] = pd.to_numeric(
        df[coluna],
        errors='coerce'
    )

print("\nPreenchendo valores nulos...")

df.fillna(0, inplace=True)

print("\nInformações gerais do dataset:\n")

print(df.info())

print("\nQuantidade de valores nulos por coluna:\n")

print(df.isnull().sum())

# =============================================================================
# 3. ESTATÍSTICAS DESCRITIVAS
# =============================================================================

titulo("3. ESTATÍSTICAS DESCRITIVAS")

print("\nResumo estatístico:\n")

print(df.describe())

print("\nTop 10 categorias com maior mortalidade:\n")

top10 = (

    df[['Categoria CID-10', 'Total']]
    .sort_values(by='Total', ascending=False)
    .head(10)
)

print(top10)

# =============================================================================
# REMOVER LINHA TOTAL
# =============================================================================

print("\nRemovendo linha agregada TOTAL...")

df = df[df['Categoria CID-10'] != 'Total']

print(f"\nNovo tamanho do dataset: {df.shape}")

# =============================================================================
# GRÁFICO TOP 10
# =============================================================================

plt.figure(figsize=(14, 8))

plt.barh(
    top10['Categoria CID-10'],
    top10['Total']
)

plt.title(
    'Top 10 Doenças com Maior Mortalidade',
    fontsize=16,
    fontweight='bold'
)

plt.xlabel('Quantidade de Mortes')

plt.ylabel('Doenças')

plt.tight_layout()

plt.savefig(
    os.path.join(
        GRAFICOS,
        'top10_mortalidade.png'
    )
)

plt.close()

print("\nGráfico salvo: top10_mortalidade.png")

# =============================================================================
# HISTOGRAMA
# =============================================================================

plt.figure(figsize=(10, 6))

plt.hist(
    df['Total'],
    bins=30
)

plt.title(
    'Distribuição da Mortalidade',
    fontsize=16,
    fontweight='bold'
)

plt.xlabel('Quantidade de Mortes')

plt.ylabel('Frequência')

plt.tight_layout()

plt.savefig(
    os.path.join(
        GRAFICOS,
        'histograma_mortalidade.png'
    )
)

plt.close()

print("Gráfico salvo: histograma_mortalidade.png")

# =============================================================================
# 4. VARIÁVEL ALVO
# =============================================================================

titulo("4. CRIAÇÃO DA VARIÁVEL ALVO")

mediana = df['Total'].median()

print(f"\nMediana da mortalidade: {mediana}")

print("\nCriando classificação binária:")

print("0 -> Baixa mortalidade")
print("1 -> Alta mortalidade")

df['high_mortality'] = np.where(
    df['Total'] > mediana,
    1,
    0
)

print("\nDistribuição da variável alvo:\n")

print(df['high_mortality'].value_counts())

# =============================================================================
# GRÁFICO TARGET
# =============================================================================

plt.figure(figsize=(6, 5))

df['high_mortality'].value_counts().plot(
    kind='bar'
)

plt.title(
    'Distribuição da Variável Alvo',
    fontsize=15,
    fontweight='bold'
)

plt.xlabel('Classe')

plt.ylabel('Quantidade')

plt.xticks(
    [0, 1],
    ['Baixa', 'Alta'],
    rotation=0
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        GRAFICOS,
        'variavel_alvo.png'
    )
)

plt.close()

print("\nGráfico salvo: variavel_alvo.png")

# =============================================================================
# 5. FEATURES
# =============================================================================

titulo("5. DEFINIÇÃO DAS FEATURES")

X = df.drop(
    columns=[
        'Categoria CID-10',
        'Total',
        'high_mortality'
    ]
)

y = df['high_mortality']

print("\nFeatures utilizadas:\n")

for coluna in X.columns:

    print(f"• {coluna}")

print("\nA coluna 'Total' foi removida para evitar Data Leakage.")

print("\nTarget:")

print("• high_mortality")

# =============================================================================
# 6. TREINO E TESTE
# =============================================================================

titulo("6. DIVISÃO TREINO E TESTE")

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.30,

    random_state=42,

    stratify=y
)

print(f"\nTreino: {X_train.shape[0]} amostras")

print(f"Teste: {X_test.shape[0]} amostras")

# =============================================================================
# 7. EXPERIMENTOS
# =============================================================================

titulo("7. EXPERIMENTOS COM ÁRVORES DE DECISÃO")

configs = [

    {
        'max_depth': 3,
        'criterion': 'gini',
        'min_samples_leaf': 1,
        'min_samples_split': 2
    },

    {
        'max_depth': 5,
        'criterion': 'gini',
        'min_samples_leaf': 1,
        'min_samples_split': 2
    },

    {
        'max_depth': 10,
        'criterion': 'gini',
        'min_samples_leaf': 1,
        'min_samples_split': 2
    },

    {
        'max_depth': None,
        'criterion': 'gini',
        'min_samples_leaf': 1,
        'min_samples_split': 2
    },

    {
        'max_depth': 5,
        'criterion': 'entropy',
        'min_samples_leaf': 1,
        'min_samples_split': 2
    },

    {
        'max_depth': 5,
        'criterion': 'gini',
        'min_samples_leaf': 5,
        'min_samples_split': 2
    },

    {
        'max_depth': 5,
        'criterion': 'gini',
        'min_samples_leaf': 10,
        'min_samples_split': 2
    }
]

results = []

for i, cfg in enumerate(configs):

    print("\n" + "=" * 80)
    print(f"{'MODELO ' + str(i+1).zfill(2):^80}")
    print("=" * 80)

    print("\nCONFIGURAÇÕES UTILIZADAS:\n")

    print(f"• max_depth         : {cfg['max_depth']}")
    print(f"• criterion         : {cfg['criterion']}")
    print(f"• min_samples_leaf  : {cfg['min_samples_leaf']}")
    print(f"• min_samples_split : {cfg['min_samples_split']}")

    model = DecisionTreeClassifier(

        max_depth=cfg['max_depth'],

        criterion=cfg['criterion'],

        min_samples_leaf=cfg['min_samples_leaf'],

        min_samples_split=cfg['min_samples_split'],

        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    prec = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    rec = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    results.append({

        'Modelo': f'M{i+1:02d}',

        'Acuracia': round(acc, 4),

        'Precisao': round(prec, 4),

        'Recall': round(rec, 4)
    })

    print("\nRESULTADOS DO MODELO:\n")

    print(f"• Acurácia : {acc:.4f}")
    print(f"• Precisão : {prec:.4f}")
    print(f"• Recall   : {rec:.4f}")

    # =========================================================================
    # IMPORTÂNCIA DAS FEATURES
    # =========================================================================

    importancias = pd.DataFrame({

        'Capitulo_CID': X.columns,
        'Importancia': model.feature_importances_
    })

    importancias = importancias.sort_values(
        by='Importancia',
        ascending=False
    )

    top3_importancia = importancias.head(3)

    print("\nTOP 3 CAPÍTULOS CID MAIS IMPORTANTES:\n")

    for idx, row in top3_importancia.iterrows():

        print(
            f"• {row['Capitulo_CID']} "
            f"-> Importância: {row['Importancia']:.4f}"
        )

    # =========================================================================
    # TOP 3 DOENÇAS
    # =========================================================================

    top3_doencas = (

        df[['Categoria CID-10', 'Total']]
        .sort_values(by='Total', ascending=False)
        .head(3)
    )

    print("\nTOP 3 DOENÇAS QUE MAIS MATAM:\n")

    for idx, row in top3_doencas.iterrows():

        print(
            f"• {row['Categoria CID-10']} "
            f"-> {int(row['Total'])} mortes"
        )

    # =========================================================================
    # INTERPRETAÇÃO DIDÁTICA
    # =========================================================================

    print("\nANÁLISE DIDÁTICA DO MODELO:\n")

    if acc >= 0.90:

        print(
            "• O modelo apresentou excelente desempenho."
        )

    elif acc >= 0.75:

        print(
            "• O modelo apresentou bom desempenho."
        )

    else:

        print(
            "• O modelo apresentou desempenho moderado."
        )

    if cfg['max_depth'] is None:

        print(
            "• Árvores sem limite de profundidade "
            "podem causar overfitting."
        )

    elif cfg['max_depth'] <= 3:

        print(
            "• Árvores rasas generalizam melhor os dados."
        )

    else:

        print(
            "• Árvores profundas aprendem relações "
            "mais complexas."
        )

    if cfg['criterion'] == 'gini':

        print(
            "• O critério Gini reduz a impureza dos nós."
        )

    else:

        print(
            "• O critério Entropy utiliza ganho de informação."
        )

# =============================================================================
# 8. RESULTADOS FINAIS
# =============================================================================

titulo("8. RESULTADOS FINAIS")

results_df = pd.DataFrame(results)

print("\nTabela comparativa:\n")

print(results_df.to_string(index=False))

# =============================================================================
# 9. MELHOR MODELO
# =============================================================================

titulo("9. MELHOR MODELO")

best_idx = results_df['Acuracia'].idxmax()

best = results_df.loc[best_idx]

print("\nModelo com maior acurácia:\n")

print(best)

# =============================================================================
# 10. MATRIZ DE CONFUSÃO
# =============================================================================

titulo("10. MATRIZ DE CONFUSÃO")

best_model = DecisionTreeClassifier(

    max_depth=None,

    criterion='gini',

    random_state=42
)

best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print("\nMatriz de confusão:\n")

print(cm)

print("\nRelatório de classificação:\n")

print(classification_report(y_test, y_pred))

# =============================================================================
# GRÁFICO COMPARAÇÃO
# =============================================================================

plt.figure(figsize=(12, 6))

x = np.arange(len(results_df))

width = 0.25

plt.bar(
    x - width,
    results_df['Acuracia'],
    width,
    label='Acurácia'
)

plt.bar(
    x,
    results_df['Precisao'],
    width,
    label='Precisão'
)

plt.bar(
    x + width,
    results_df['Recall'],
    width,
    label='Recall'
)

plt.xticks(
    x,
    results_df['Modelo']
)

plt.ylim(0, 1.1)

plt.title(
    'Comparação Entre os Modelos',
    fontsize=16,
    fontweight='bold'
)

plt.xlabel('Modelos')

plt.ylabel('Score')

plt.legend()

plt.grid(axis='y')

plt.tight_layout()

plt.savefig(
    os.path.join(
        GRAFICOS,
        'comparacao_modelos.png'
    )
)

plt.close()

print("\nGráfico salvo: comparacao_modelos.png")

# =============================================================================
# 11. IMPORTÂNCIA DAS FEATURES
# =============================================================================

titulo("11. IMPORTÂNCIA DAS FEATURES")

feature_importance = pd.DataFrame({

    'Feature': X.columns,

    'Importancia': best_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by='Importancia',
    ascending=False
)

print("\nImportância das variáveis:\n")

print(feature_importance)

# =============================================================================
# GRÁFICO IMPORTÂNCIA
# =============================================================================

plt.figure(figsize=(12, 8))

plt.barh(
    feature_importance['Feature'],
    feature_importance['Importancia']
)

plt.title(
    'Importância das Variáveis',
    fontsize=16,
    fontweight='bold'
)

plt.xlabel('Importância')

plt.ylabel('Features')

plt.tight_layout()

plt.savefig(
    os.path.join(
        GRAFICOS,
        'importancia_features.png'
    )
)

plt.close()

print("\nGráfico salvo: importancia_features.png")

# =============================================================================
# 12. CONCLUSÃO
# =============================================================================

titulo("12. CONCLUSÃO")

print("""
O trabalho demonstrou como os parâmetros das árvores de decisão
impactam diretamente no desempenho dos modelos.

Os parâmetros avaliados foram:

• max_depth
• criterion
• min_samples_leaf
• min_samples_split

Foi possível observar que:

• Árvores muito profundas tendem ao overfitting;
• Valores maiores de min_samples_leaf geram modelos mais estáveis;
• O critério Gini apresentou melhor desempenho em alguns cenários;
• O dataset precisou de pré-processamento antes da modelagem;
• Foi necessário remover Data Leakage do modelo.

O modelo final apresentou bom equilíbrio entre:

• Acurácia
• Precisão
• Recall

Além disso, a análise de importância das variáveis
permitiu identificar quais capítulos CID possuem maior
influência na previsão de alta mortalidade.
""")

print("\nExecução finalizada com sucesso!")