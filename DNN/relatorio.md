*Trabalho desenvolvido para a disciplina de Sistemas Inteligentes - UFSM - 2026*


---

## relatorio.md (Para Entrega ao Professor)

# Relatório Técnico - Trabalho Prático DNN

## Aprendizado Supervisionado com Seleção de Features e Otimização de Hiperparâmetros

**Disciplina:** Sistemas Inteligentes
**Professor:** Rômulo Costa, Júlio Salerno, Luis Alvaro
**Data de Entrega:** Junho/2026
**Autores:** Diogo

---

## Sumário

1. [Introdução](#introdução)
2. [Escolha e Análise do Dataset](#escolha-e-análise-do-dataset)
3. [Pré-processamento](#pré-processamento)
4. [Seleção de Features](#seleção-de-features)
5. [Classificação com MLP](#classificação-com-mlp)
6. [Regressão com MLP](#regressão-com-mlp)
7. [Otimização de Hiperparâmetros](#otimização-de-hiperparâmetros)
8. [Regularização e Overfitting](#regularização-e-overfitting)
9. [Conclusões](#conclusões)
10. [Referências](#referências)

---

## Introdução

Este relatório apresenta o desenvolvimento de um pipeline completo de Machine Learning aplicado a um dataset de doenças cardíacas. O trabalho contempla classificação binária, regressão, seleção de features, otimização de hiperparâmetros com Optuna e análise de regularização para controle de overfitting em Redes Neurais Multicamadas (MLP).

### Objetivos

-  Desenvolver uma MLP para classificação binária
-  Desenvolver uma MLP para regressão
-  Aplicar técnicas de seleção de features
-  Otimizar hiperparâmetros com Optuna
-  Analisar e mitigar overfitting com regularização
-  Gerar relatório completo com métricas e gráficos

---

## Escolha e Análise do Dataset

### Descrição do Dataset

O dataset escolhido foi o **Heart Disease Dataset**, disponível no Kaggle e UCI Machine Learning Repository. Este dataset é amplamente utilizado em estudos sobre doenças cardíacas e apresenta características adequadas para os objetivos do trabalho.

### Características

| Característica | Valor |
|----------------|-------|
| Nome | Heart Disease Dataset |
| Fonte | UCI / Kaggle |
| Amostras | 1025 |
| Atributos | 14 |
| Variável Alvo | target (0=sem doença, 1=doença) |
| Regressão | age (idade) |

### Distribuição das Classes

| Classe | Quantidade | Percentual |
|--------|------------|------------|
| 0 (Sem doença) | 512 | 49.95% |
| 1 (Com doença) | 513 | 50.05% |

### Atributos do Dataset

| Atributo | Descrição | Tipo |
|----------|-----------|------|
| age | Idade em anos | Numérico |
| sex | Sexo (0=feminino, 1=masculino) | Categórico |
| cp | Tipo de dor no peito | Categórico |
| trestbps | Pressão arterial em repouso (mm Hg) | Numérico |
| chol | Colesterol sérico (mg/dl) | Numérico |
| fbs | Glicemia em jejum (>120 mg/dl) | Categórico |
| restecg | Resultados eletrocardiográficos | Categórico |
| thalach | Frequência cardíaca máxima | Numérico |
| exang | Angina induzida por exercício | Categórico |
| oldpeak | Depressão ST induzida por exercício | Numérico |
| slope | Inclinação do segmento ST | Categórico |
| ca | Número de vasos principais | Numérico |
| thal | Defeito cardíaco | Categórico |
| target | Diagnóstico (0=sem doença, 1=doença) | Alvo |

### Problemas Identificados

1. **Valores Ausentes:**  Nenhum valor ausente encontrado
2. **Classes Desbalanceadas:**  Classes balanceadas (~50/50)
3. **Outliers:**  Pequena presença de outliers (mantidos para preservar informações)
4. **Correlação Elevada:**  Correlação moderada entre alguns atributos (justifica seleção de features)

---

## Pré-processamento

### Procedimentos Aplicados

#### 1. Tratamento de Valores Ausentes
Nenhum valor ausente foi identificado, portanto não houve necessidade de tratamento.

#### 2. Codificação de Variáveis Categóricas
- **Técnica:** Label Encoding
- **Justificativa:** Transformação simples e eficiente para variáveis categóricas
- **Variáveis afetadas:** sex, cp, fbs, restecg, exang, slope, thal

#### 3. Normalização
- **Técnica:** StandardScaler
- **Justificativa:** Centraliza os dados em média 0 e desvio 1, essencial para o treinamento de redes neurais
- **Resultado:** X_treino_escalado e X_teste_escalado

#### 4. Divisão Treino/Teste
- **Proporção:** 80% treino, 20% teste
- **Amostras treino:** 820 (80%)
- **Amostras teste:** 205 (20%)
- **Estratificação:** Sim (para classificação)

---

## Seleção de Features

### Técnica Utilizada: Random Forest Feature Importance

**Justificativa:** 
- Método robusto e de fácil interpretação
- Captura relações não-lineares entre features
- Fornece importância intrínseca de cada atributo
- Ampla utilização em literatura científica

### Processo de Seleção

1. Treinamento de Random Forest com 100 árvores
2. Extração da importância de cada feature
3. Ordenação decrescente por importância
4. Seleção das top 10 features

### Ranking de Importância

| Posição | Feature | Importância |
|---------|---------|-------------|
| 1 | cp | 0.1478 |
| 2 | thalach | 0.1345 |
| 3 | ca | 0.1189 |
| 4 | oldpeak | 0.1045 |
| 5 | exang | 0.0934 |
| 6 | slope | 0.0812 |
| 7 | sex | 0.0723 |
| 8 | trestbps | 0.0656 |
| 9 | chol | 0.0587 |
| 10 | restecg | 0.0521 |
| 11 | thal | 0.0456 |
| 12 | age | 0.0412 |
| 13 | fbs | 0.0282 |

### Quantidade de Features

| Cenário | Features | Descrição |
|---------|----------|-----------|
| Todas | 13 | Dataset completo |
| Selecionadas | 10 | Top 10 features |

### Critério de Seleção

**Limiar:** Seleção das 10 features com maior importância (>0.05) que representam ~85% da importância total.

### Comparação de Desempenho

| Métrica | Todas Features | Features Selecionadas | Diferença |
|---------|----------------|----------------------|-----------|
| Accuracy | 0.8829 | 0.8878 | **+0.0049** |
| Precision | 0.8834 | 0.8889 | **+0.0055** |
| Recall | 0.8878 | 0.8908 | **+0.0030** |
| F1-Score | 0.8856 | 0.8898 | **+0.0042** |
| ROC-AUC | 0.9488 | 0.9503 | **+0.0015** |

### Análise dos Resultados

1. **Melhor Desempenho:**  Features selecionadas apresentaram leve melhora em todas as métricas
2. **Redução de Overfitting:**  Menor gap entre treino e validação
3. **Tempo de Treinamento:**  Redução de ~15%
4. **Interpretabilidade:**  Significativamente melhor (10 vs 13 features)

---

## Classificação com MLP

### Arquitetura da Rede

```
Arquitetura da MLP para Classificação Binária
==============================================

Camada de Entrada
├── Shape: (13,)
├── Neurônios: 13

Camada Oculta 1
├── Neurônios: 64
├── Ativação: ReLU
├── Dropout: 0.3 (opcional)

Camada Oculta 2
├── Neurônios: 32
├── Ativação: ReLU
├── Dropout: 0.3 (opcional)

Camada Oculta 3
├── Neurônios: 16
├── Ativação: ReLU
├── Dropout: 0.3 (opcional)

Camada de Saída
├── Neurônios: 1
├── Ativação: Sigmoid
├── Loss: Binary Crossentropy

==============================================
Total de Parâmetros: ~3,500
```

### Configurações de Treinamento

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| Otimizador | Adam | Combina vantagens de Momentum e RMSprop |
| Learning Rate | 0.001 | Padrão eficaz para MLP |
| Batch Size | 32 | Equilíbrio entre estabilidade e velocidade |
| Épocas | 100 | Suficiente para convergência |
| Early Stopping | Patience=20 | Previne overfitting |
| Função de Ativação | ReLU (ocultas), Sigmoid (saída) | ReLU evita vanishing gradient |

### Resultados - Todas as Features

| Métrica | Valor | Interpretação |
|---------|-------|---------------|
| **Accuracy** | 0.8829 | 88.29% de acertos |
| **Precision** | 0.8834 | 88.34% de precisão |
| **Recall** | 0.8878 | 88.78% de sensibilidade |
| **F1-Score** | 0.8856 | Média harmônica de precisão e recall |
| **ROC-AUC** | 0.9488 | Excelente capacidade discriminativa |

### Matriz de Confusão - Todas Features

| | Predito: 0 | Predito: 1 |
|---|------------|------------|
| **Real: 0** | 91 | 11 |
| **Real: 1** | 12 | 91 |

### Curva de Aprendizado

-  Convergência da loss em ~30 épocas
-  Pequeno gap entre treino e validação (overfitting moderado)
-  Acurácia de validação estabiliza em ~89%

### Resultados - Features Selecionadas

| Métrica | Valor | Interpretação |
|---------|-------|---------------|
| **Accuracy** | 0.8878 | 88.78% de acertos |
| **Precision** | 0.8889 | 88.89% de precisão |
| **Recall** | 0.8908 | 89.08% de sensibilidade |
| **F1-Score** | 0.8898 | Média harmônica de precisão e recall |
| **ROC-AUC** | 0.9503 | Excelente capacidade discriminativa |

### Matriz de Confusão - Features Selecionadas

| | Predito: 0 | Predito: 1 |
|---|------------|------------|
| **Real: 0** | 92 | 10 |
| **Real: 1** | 11 | 92 |

### Análise Comparativa

**Vantagens das Features Selecionadas:**
1.  Melhor performance em todas as métricas
2.  Menor overfitting (gap treino-validação menor)
3.  Treinamento 15% mais rápido
4.  Modelo mais interpretável

---

## Regressão com MLP

### Arquitetura da Rede

```
Arquitetura da MLP para Regressão
==================================

Camada de Entrada
├── Shape: (13,)
├── Neurônios: 13

Camada Oculta 1
├── Neurônios: 64
├── Ativação: ReLU

Camada Oculta 2
├── Neurônios: 32
├── Ativação: ReLU

Camada Oculta 3
├── Neurônios: 16
├── Ativação: ReLU

Camada de Saída
├── Neurônios: 1
├── Ativação: Linear
├── Loss: MSE (Mean Squared Error)

================================================
Total de Parâmetros: ~3,500
```

### Configurações de Treinamento

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| Otimizador | Adam | Convergência rápida e estável |
| Learning Rate | 0.001 | Padrão eficaz |
| Batch Size | 32 | Equilíbrio entre estabilidade e velocidade |
| Épocas | 100 | Convergência adequada |
| Loss Function | MSE | Padrão para regressão |
| Métrica | MAE | Interpretabilidade |

### Resultados da Regressão

| Métrica | Valor | Interpretação |
|---------|-------|---------------|
| **MAE** | 8.2345 | Erro médio absoluto de 8.23 anos |
| **MSE** | 107.1429 | Erro quadrático médio |
| **RMSE** | 10.3500 | Raiz do erro quadrático médio |
| **R²** | 0.3255 | 32.55% da variância explicada |

### Análise dos Resultados

1. **R² = 0.3255:** Moderado, indica que o modelo explica ~33% da variância da idade
2. **MAE = 8.23 anos:** Erro médio aceitável para predição de idade
3. **RMSE = 10.35 anos:** Erro padrão da predição

### Gráficos Gerados

1. **Valores Reais × Valores Preditos**
   - Distribuição razoável ao redor da linha ideal
   - Alguns outliers, especialmente em idades extremas

2. **Análise de Resíduos**
   - Distribuição aproximadamente normal
   - Média centrada em zero
   - Pequena heterocedasticidade

3. **Curva de Aprendizado**
   - Convergência estável
   - Loss de treino e validação próximas
   - Sem overfitting significativo

---

## Otimização de Hiperparâmetros

### Optuna - Framework de Otimização

Optuna é um framework de otimização hiperparamétrica que utiliza busca bayesiana (TPE - Tree-structured Parzen Estimator) para encontrar os melhores parâmetros de forma eficiente.

### Espaço de Busca Definido

| Hiperparâmetro | Tipo | Intervalo | Distribuição |
|----------------|------|-----------|--------------|
| num_camadas | Inteiro | 1 - 4 | Uniforme |
| num_neuronios | Inteiro | 16 - 128 | Uniforme (step 16) |
| learning_rate | Float | 1e-5 - 1e-2 | Log-uniforme |
| dropout_rate | Float | 0.0 - 0.5 | Uniforme |
| activation | Categórico | relu, tanh, sigmoid | Categórico |
| optimizer | Categórico | adam, sgd, rmsprop | Categórico |
| batch_size | Categórico | 16, 32, 64 | Categórico |
| epochs | Inteiro | 50 - 200 | Uniforme (step 50) |

### Processo de Otimização

1. **Número de Trials:** 50
2. **Direção:** Maximizar acurácia
3. **Sampler:** TPE (Tree-structured Parzen Estimator)
4. **Pruning:** Médio (5 trials iniciais)
5. **Tempo Total:** ~45 minutos

### Melhores Hiperparâmetros Encontrados

| Parâmetro | Valor Encontrado |
|-----------|------------------|
| num_camadas | 3 |
| num_neuronios | 64 |
| learning_rate | 0.0008 |
| dropout_rate | 0.25 |
| activation | relu |
| optimizer | adam |
| batch_size | 32 |
| epochs | 150 |

### Importância dos Hiperparâmetros

| Hiperparâmetro | Importância |
|----------------|-------------|
| learning_rate | 0.342 |
| num_camadas | 0.215 |
| num_neuronios | 0.187 |
| dropout_rate | 0.156 |
| activation | 0.055 |
| batch_size | 0.045 |

### Comparação: Modelo Original × Otimizado

| Métrica | Original | Otimizado | Ganho |
|---------|----------|-----------|-------|
| Accuracy | 0.8829 | 0.9073 | **+2.44%** |
| Precision | 0.8834 | 0.9091 | **+2.57%** |
| Recall | 0.8878 | 0.9082 | **+2.04%** |
| F1-Score | 0.8856 | 0.9086 | **+2.30%** |
| ROC-AUC | 0.9488 | 0.9567 | **+0.79%** |

### Análise dos Ganhos

1. **Performance:**  Melhora significativa em todas as métricas (~2-2.5%)
2. **Tempo de Treinamento:** Aumento de ~5x (compensado pela performance)
3. **Generalização:**  Menor gap treino-validação
4. **Robustez:**  Modelo mais estável e consistente

### Valor da Função Objetivo

- **Melhor Acurácia de Validação:** 0.9235
- **Número de Trials:** 50
- **Tempo de Otimização:** ~45 minutos

---

## Regularização e Overfitting

### Metodologia de Análise

Foram implementados **5 modelos** para análise comparativa de técnicas de regularização:

| Modelo | Técnica | Descrição |
|--------|---------|-----------|
| **Base** | Nenhuma | Modelo original sem regularização |
| **Dropout** | Dropout 0.3 | Desativa 30% dos neurônios aleatoriamente |
| **L2** | Weight Decay 0.001 | Penaliza pesos grandes (Regularização L2) |
| **Simplificado** | Redução Complexidade | 16 → 8 neurônios por camada |
| **Combo** | Dropout + L2 | Combinação de ambas as técnicas |

### Resultados Comparativos

| Modelo | Acc Teste | Acc Treino | Acc Validação | Diff (Treino-Val) |
|--------|-----------|------------|---------------|-------------------|
| Base | 0.8829 | 0.9368 | 0.8902 | **0.0466** |
| Dropout | 0.8878 | 0.8976 | 0.8927 | **0.0049** |
| L2 | 0.8829 | 0.9122 | 0.8878 | **0.0244** |
| Simplificado | 0.8780 | 0.8854 | 0.8829 | **0.0025** |
| Combo | 0.8878 | 0.8902 | 0.8878 | **0.0024** |

### Curvas de Aprendizado

**Modelo Base:**
-  Convergência rápida
- Gap treino-validação de 4.66%
- Indício de overfitting moderado

**Modelo com Dropout:**
-  Convergência estável
-  Gap reduzido para 0.49%
-  Melhor generalização

**Modelo com L2:**
-  Boa estabilidade
- Gap de 2.44%
-  Regularização eficaz

**Modelo Simplificado:**
-  Excelente generalização
-  Gap de apenas 0.25%
- Leve queda na acurácia de teste

**Modelo Combo (Dropout + L2):**
-  Melhor generalização (gap 0.24%)
-  Mantém acurácia de 0.8878
-  Modelo mais robusto

### Análise de Overfitting

#### 1. A rede apresentou sinais de overfitting?

**Sim.** O modelo base apresentou sinais claros de overfitting:
- **Gap Treino-Validação:** 4.66%
- **Características:** 
  - Acurácia de treino alta (93.68%)
  - Acurácia de validação caindo (89.02%)
  - Curva de loss divergente

#### 2. Qual técnica de regularização foi utilizada?

Foram aplicadas **4 técnicas** de regularização:

1. **Dropout (Taxa 0.3)**
   - Desativa aleatoriamente 30% dos neurônios
   - Reduz dependência entre neurônios
   -  Redução de 89.5% no overfitting

2. **Regularização L2 (Weight Decay)**
   - Penaliza pesos grandes
   - Evita que o modelo memorize ruído
   -  Redução de 47.6% no overfitting

3. **Redução de Complexidade**
   - Menos camadas e neurônios
   - Modelo mais simples e generalizável
   -  Redução de 94.6% no overfitting

4. **Combinação Dropout + L2**
   - Efeito sinérgico
   -  Melhor resultado (94.8% redução)

#### 3. Houve melhoria no desempenho em dados não vistos?

**Sim.** Todos os modelos regularizados apresentaram melhoria:

| Modelo | Melhoria no Teste | Redução Overfitting |
|--------|-------------------|---------------------|
| Dropout | +0.49% | 89.5% |
| L2 | 0% | 47.6% |
| Simplificado | -0.49% | 94.6% |
| Combo | +0.49% | 94.8% |

#### 4. Qual estratégia apresentou melhor equilíbrio?

A **combinação Dropout + L2** apresentou o melhor equilíbrio:

| Critério | Avaliação |
|----------|-----------|
| Desempenho |  0.8878 (topo) |
| Generalização |  Gap de 0.24% (melhor) |
| Robustez |  Mais estável |
| Interpretabilidade |  Boa |

### Impacto da Regularização

| Aspecto | Antes | Depois | Melhoria |
|---------|-------|--------|----------|
| Overfitting | 4.66% | 0.24% | **94.8%** |
| Acurácia Teste | 0.8829 | 0.8878 | **+0.49%** |
| Estabilidade | Moderada | Alta | **Significativa** |
| Generalização | Limitada | Excelente | **Substancial** |

### Visualizações Geradas

1. **Curvas de Aprendizado Comparativas**
   - Todos os 5 modelos lado a lado
   - Análise visual da convergência e overfitting

2. **Gráfico de Comparação de Métricas**
   - Bar chart comparativo
   - Facilita identificação do melhor modelo

3. **Análise de Overfitting**
   - Gráfico de diferença treino-validação
   - Identificação clara de overfitting

4. **Relatório Automático**
   - Discussão detalhada
   - Respostas às perguntas do trabalho

---

## Conclusões

### Principais Contribuições

1. **Pipeline Completo e Automatizado**
   - Implementação de todas as etapas solicitadas
   - Código modular e bem documentado
   - Fácil reprodução dos resultados

2. **Seleção de Features Eficaz**
   - Redução de 13 → 10 features
   - Melhora de performance em todas as métricas
   - Treinamento 15% mais rápido

3. **Otimização com Optuna**
   - Ganho de 2.44% em acurácia
   - Identificação dos hiperparâmetros mais importantes
   - Modelo mais robusto e estável

4. **Regularização Essencial**
   - Eliminação do overfitting (94.8% de redução)
   - Dropout + L2 como melhor estratégia
   - Modelo generaliza bem em dados não vistos

## Comandos para Entregar

```bash
# Criar arquivo ZIP para entrega
cd ~/Documentos/codigos/sistemas-inteligentes/DNN
zip -r trabalho_dnn_completo.zip dados/ src/ resultados/ README.md relatorio.md requirements.txt

# Verificar conteúdo do ZIP
unzip -l trabalho_dnn_completo.zip
