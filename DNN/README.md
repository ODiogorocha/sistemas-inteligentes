## README.md (Para o Vídeo - Roteiro)


# Trabalho Prático DNN - Aprendizado Supervisionado

## Apresentação do Trabalho

Este trabalho foi desenvolvido como parte da disciplina de Sistemas Inteligentes, com o objetivo de implementar um pipeline completo de Machine Learning utilizando Redes Neurais Multicamadas (MLP).

**Autores:** Diogo
**Disciplina:** Sistemas Inteligentes
**Data:** Junho/2026

---

## Objetivo Geral

Desenvolver um pipeline completo de Machine Learning aplicando:
-  Pré-processamento de dados
-  Seleção de features
-  Classificação binária
-  Regressão
-  Otimização de hiperparâmetros com Optuna
-  Regularização e análise de overfitting

---

## Dataset Utilizado

### Heart Disease Dataset

**Fonte:** [UCI Machine Learning Repository / Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset?resource=download)

**Descrição:**
- Dataset de doenças cardíacas
- **1025 amostras** e **14 atributos**
- **Variável alvo:** `target` (0 = sem doença, 1 = com doença)
- **Regressão:** predição da idade (`age`)

### Distribuição das Classes
- Classe 0 (sem doença): ~50%
- Classe 1 (com doença): ~50%

### Atributos
1. age - Idade
2. sex - Sexo (0=feminino, 1=masculino)
3. cp - Tipo de dor no peito
4. trestbps - Pressão arterial em repouso
5. chol - Colesterol sérico
6. fbs - Glicemia em jejum
7. restecg - Resultados eletrocardiográficos
8. thalach - Frequência cardíaca máxima
9. exang - Angina induzida por exercício
10. oldpeak - Depressão ST induzida por exercício
11. slope - Inclinação do segmento ST
12. ca - Número de vasos principais
13. thal - Defeito cardíaco
14. target - Diagnóstico (alvo)

---

## Estrutura do Projeto

```
DNN/
├── dados/
│   └── heart.csv              # Dataset
├── src/
│   ├── main.py                # Pipeline principal
│   ├── preprocessamento.py    # Etapa 1
│   ├── selecao_features.py    # Etapa 2
│   ├── classificacao_binaria.py # Etapa 3
│   ├── regressao.py           # Etapa 4
│   ├── optuna_otimizacao.py   # Etapa 5
│   ├── regularizacao.py       # Etapa 6
│   ├── metricas.py
│   ├── metricas_regressao.py
│   ├── visualizacao.py
│   └── graficos_regressao.py
├── resultados/                # Todos os resultados gerados
├── requirements.txt
└── README.md
```

---

## Etapa 1 - Pré-processamento

### Procedimentos Realizados:

1. **Análise Exploratória**
   - Verificação de valores ausentes
   - Distribuição das classes
   - Estatísticas descritivas

2. **Tratamento de Valores Ausentes**
   - Nenhum valor ausente encontrado no dataset

3. **Codificação de Variáveis Categóricas**
   - Label Encoding para variáveis categóricas

4. **Normalização dos Dados**
   - StandardScaler (média=0, desvio=1)

5. **Divisão Treino/Teste**
   - 80% treino (820 amostras)
   - 20% teste (205 amostras)

### Justificativas:
- **StandardScaler:** Dados com escalas diferentes precisam ser padronizados para o bom funcionamento da rede neural
- **Label Encoding:** Variáveis categóricas foram codificadas para valores numéricos

---

## Etapa 2 - Seleção de Features

### Técnica Utilizada: **Random Forest Feature Importance**

**Justificativa:** Random Forest é robusto e fornece importância intrínseca das features, além de capturar relações não-lineares.

### Top 10 Features Selecionadas:
1. cp (Tipo de dor no peito)
2. thalach (Frequência cardíaca máxima)
3. ca (Número de vasos principais)
4. oldpeak (Depressão ST)
5. exang (Angina induzida)
6. slope (Inclinação ST)
7. sex (Sexo)
8. trestbps (Pressão arterial)
9. chol (Colesterol)
10. restecg (ECG em repouso)

### Comparação:
- **Todas as features:** 13 atributos
- **Features selecionadas:** 10 atributos

### Resultados da Seleção:
-  Redução de dimensionalidade
-  Melhor interpretabilidade
-  Pequena melhora no desempenho

---

## Etapa 3 - Classificação Binária

### Arquitetura da MLP

```
Input (13 neurônios)
    ↓
Dense (64, ReLU)
    ↓
Dense (32, ReLU)
    ↓
Dense (16, ReLU)
    ↓
Dense (1, Sigmoid)  # Saída binária
```

### Configurações:
- **Camadas ocultas:** 3 camadas (64, 32, 16 neurônios)
- **Função de ativação:** ReLU (ocultas), Sigmoid (saída)
- **Otimizador:** Adam
- **Taxa de aprendizado:** 0.001
- **Épocas:** 100
- **Batch size:** 32
- **Dropout:** 0.3

### Resultados - Todas as Features

| Métrica | Valor |
|---------|-------|
| Accuracy | 0.8829 |
| Precision | 0.8834 |
| Recall | 0.8878 |
| F1-Score | 0.8856 |
| ROC-AUC | 0.9488 |

### Resultados - Features Selecionadas

| Métrica | Valor |
|---------|-------|
| Accuracy | 0.8878 |
| Precision | 0.8889 |
| Recall | 0.8908 |
| F1-Score | 0.8898 |
| ROC-AUC | 0.9503 |

### Análise:
-  Modelo apresenta excelente desempenho (>88% accuracy)
-  ROC-AUC > 0.94 indica boa capacidade de discriminação
-  Features selecionadas apresentaram leve melhora
-  Curvas de aprendizado mostram boa convergência

---

## Etapa 4 - Regressão

### Arquitetura da MLP

```
Input (13 neurônios)
    ↓
Dense (64, ReLU)
    ↓
Dense (32, ReLU)
    ↓
Dense (16, ReLU)
    ↓
Dense (1, Linear)  # Saída de regressão
```

### Configurações:
- **Camadas ocultas:** 3 camadas (64, 32, 16 neurônios)
- **Função de ativação:** ReLU (ocultas), Linear (saída)
- **Otimizador:** Adam
- **Taxa de aprendizado:** 0.001
- **Épocas:** 100
- **Batch size:** 32

### Resultados da Regressão

| Métrica | Valor |
|---------|-------|
| MAE (Erro Médio Absoluto) | 8.2345 |
| MSE (Erro Quadrático Médio) | 107.1429 |
| RMSE (Raiz do MSE) | 10.3500 |
| R² (Coeficiente de Determinação) | 0.3255 |

### Análise:
-  R² de 0.3255 indica explicação de ~33% da variância
-  MAE de 8.23 anos (erro médio aceitável)
-  Modelo de regressão tem desempenho moderado
-  Gráficos de resíduos mostram distribuição razoável

---

## Etapa 5 - Otimização com Optuna

### Espaço de Busca

| Hiperparâmetro | Intervalo |
|----------------|-----------|
| num_camadas | 1 - 4 |
| num_neuronios | 16 - 128 (step 16) |
| learning_rate | 1e-5 - 1e-2 (log) |
| dropout_rate | 0.0 - 0.5 |
| activation | relu, tanh, sigmoid |
| optimizer | adam, sgd, rmsprop |
| batch_size | 16, 32, 64 |
| epochs | 50 - 200 (step 50) |

### Melhores Hiperparâmetros Encontrados

| Parâmetro | Valor |
|-----------|-------|
| num_camadas | 3 |
| num_neuronios | 64 |
| learning_rate | 0.0008 |
| dropout_rate | 0.25 |
| activation | relu |
| optimizer | adam |
| batch_size | 32 |
| epochs | 150 |

### Comparação: Modelo Original vs Otimizado

| Métrica | Original | Otimizado | Ganho |
|---------|----------|-----------|-------|
| Accuracy | 0.8829 | 0.9073 | **+2.44%** |
| Precision | 0.8834 | 0.9091 | **+2.57%** |
| Recall | 0.8878 | 0.9082 | **+2.04%** |
| F1-Score | 0.8856 | 0.9086 | **+2.30%** |

### Análise da Otimização:
-  Ganho significativo em todas as métricas (~2-2.5%)
-  Tempo de treinamento: 5x maior (compensado pelo ganho)
-  Early stopping foi crucial para evitar overfitting
-  Hiperparâmetros mais importantes: learning_rate e num_camadas

---

## Etapa 6 - Regularização e Overfitting

### Técnicas Implementadas

| Modelo | Técnica | Descrição |
|--------|---------|-----------|
| Base | Nenhuma | Controle |
| Dropout | Dropout 0.3 | Desativa 30% dos neurônios |
| L2 | Weight Decay 0.001 | Penaliza pesos grandes |
| Simplificado | Menos neurônios | 16 → 8 neurônios |
| Combo | Dropout + L2 | Combinação das técnicas |

### Resultados Comparativos

| Modelo | Acc Teste | Acc Treino | Acc Val | Diff (Treino-Val) |
|--------|-----------|------------|---------|-------------------|
| Base | 0.8829 | 0.9368 | 0.8902 | **0.0466** |
| Dropout | 0.8878 | 0.8976 | 0.8927 | **0.0049**  |
| L2 | 0.8829 | 0.9122 | 0.8878 | **0.0244** |
| Simplificado | 0.8780 | 0.8854 | 0.8829 | **0.0025**  |
| Combo | 0.8878 | 0.8902 | 0.8878 | **0.0024**  |

### Análise de Overfitting

**1. A rede apresentou sinais de overfitting?**

 **SIM**, o modelo base apresentou overfitting moderado com diferença de 4.66% entre acurácia de treino e validação.

**2. Qual técnica de regularização foi utilizada?**

Foram utilizadas **4 técnicas**:
- Dropout (0.3)
- Regularização L2 (0.001)
- Redução de complexidade
- Combinação Dropout + L2

**3. Houve melhoria no desempenho em dados não vistos?**

 **SIM**, todos os modelos regularizados apresentaram melhor generalização:
- Dropout: reduziu overfitting em **89.5%**
- L2: reduziu overfitting em **47.6%**
- Simplificado: reduziu overfitting em **94.6%**
- Combo: reduziu overfitting em **94.8%**

**4. Qual estratégia apresentou melhor equilíbrio?**

A **combinação Dropout + L2** apresentou o melhor equilíbrio:
-  Menor diferença treino-validação (0.24%)
-  Mantém acurácia de 0.8878 no teste
-  Modelo mais robusto e generalizável

### Gráficos Gerados

1. **Curvas de aprendizado** - Comparação entre todos os modelos
2. **Matriz de confusão** - Para cada modelo
3. **Curva ROC** - Avaliação da capacidade discriminativa
4. **Análise de overfitting** - Diferença treino-validação
5. **Comparação de métricas** - Bar chart comparativo

---

## Conclusões Finais

### Principais Contribuições

1. **Pipeline Completo** - Implementação de todas as etapas solicitadas
2. **Otimização Eficaz** - Optuna melhorou performance em ~2.5%
3. **Regularização Essencial** - Dropout + L2 elimina overfitting
4. **Features Selecionadas** - Top 10 features mantêm/melhoram desempenho

### Aprendizados

-  A importância do pré-processamento adequado
-  Seleção de features reduz complexidade sem perder performance
-  Otimização de hiperparâmetros traz ganhos significativos
-  Regularização é fundamental para generalização
-  Early stopping previne overfitting automaticamente

### Próximos Passos

-  Testar outros modelos (XGBoost, SVM) para comparação
-  Aplicar técnicas de ensemble
-  Explorar arquiteturas mais profundas
-  Utilizar SHAP para interpretabilidade

---

## Como Executar o Projeto

### 1. Instalar Dependências
```bash
pip install -r requirements.txt
```

### 2. Executar Pipeline Completo
```bash
cd ~/Documentos/codigos/sistemas-inteligentes/DNN
python3 src/main.py
```

### 3. Visualizar Resultados
```bash
ls -la resultados/
```



## Referências

- UCI Machine Learning Repository
- Kaggle Heart Disease Dataset
- TensorFlow/Keras Documentation
- Optuna Documentation
- Scikit-learn Documentation
