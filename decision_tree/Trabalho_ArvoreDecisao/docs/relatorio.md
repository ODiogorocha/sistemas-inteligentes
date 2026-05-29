# Universidade Federal de Santa Maria

## Centro de Tecnologia

# Trabalho Prático

# Construção e Análise de Árvores de Decisão

---

**Disciplina:** Sistemas Inteligentes
**Dataset:** Mortalidade por Categoria CID-10 no Brasil (SIM/DATASUS)

---

# 1. Introdução

Este relatório documenta o desenvolvimento e avaliação de modelos de Árvore de Decisão aplicados à análise de mortalidade no Brasil utilizando dados do Sistema de Informações sobre Mortalidade (SIM/DATASUS). O trabalho foi implementado em Python utilizando a biblioteca Scikit-Learn.

O objetivo principal é investigar como diferentes hiperparâmetros das Árvores de Decisão influenciam o desempenho do modelo na tarefa de classificação de categorias CID-10 em níveis de alta ou baixa mortalidade.

Os parâmetros analisados foram:

* `max_depth`
* `criterion`
* `min_samples_leaf`
* `min_samples_split`

O desempenho dos modelos foi avaliado utilizando as métricas:

* Acurácia
* Precisão
* Recall

Além disso, o trabalho também realiza uma análise estatística descritiva do dataset e identifica as doenças com maior número de óbitos no Brasil.

---

# 2. Dataset

O dataset utilizado foi obtido a partir do DATASUS/SIM (Sistema de Informações sobre Mortalidade), contendo registros de óbitos por categorias CID-10 no Brasil.

O conjunto de dados apresenta informações agregadas de mortalidade divididas por capítulos da Classificação Internacional de Doenças (CID-10).

O dataset contém:

* 1386 registros válidos
* 21 colunas
* Informações numéricas de mortalidade por capítulos CID
* Total geral de óbitos por categoria

---

# 2.1 Atributos

| Atributo         | Tipo     | Descrição                          |
| ---------------- | -------- | ---------------------------------- |
| Categoria CID-10 | Texto    | Nome da categoria da doença        |
| Cap I            | Numérico | Doenças infecciosas e parasitárias |
| Cap II           | Numérico | Neoplasias                         |
| Cap III          | Numérico | Doenças do sangue                  |
| Cap IV           | Numérico | Doenças endócrinas                 |
| Cap V            | Numérico | Transtornos mentais                |
| Cap VI           | Numérico | Doenças do sistema nervoso         |
| Cap VII          | Numérico | Doenças dos olhos                  |
| Cap VIII         | Numérico | Doenças dos ouvidos                |
| Cap IX           | Numérico | Doenças do aparelho circulatório   |
| Cap X            | Numérico | Doenças respiratórias              |
| Cap XI           | Numérico | Doenças digestivas                 |
| Cap XII          | Numérico | Doenças da pele                    |
| Cap XIII         | Numérico | Doenças osteomusculares            |
| Cap XIV          | Numérico | Doenças geniturinárias             |
| Cap XV           | Numérico | Gravidez e parto                   |
| Cap XVI          | Numérico | Afecções perinatais                |
| Cap XVII         | Numérico | Malformações congênitas            |
| Cap XVIII        | Numérico | Causas mal definidas               |
| Cap XX           | Numérico | Causas externas                    |
| Total            | Numérico | Total geral de óbitos              |

---

# 2.2 Pré-processamento

O dataset passou por diversas etapas de limpeza e tratamento dos dados.

As principais etapas realizadas foram:

* Remoção de linhas de metadados do DATASUS
* Remoção de colunas vazias
* Conversão de caracteres inválidos (“-”) para valores numéricos
* Conversão de strings numéricas para float
* Remoção de valores ausentes
* Remoção da linha agregada “Total”

Também foi criada uma variável alvo binária chamada:

`high_mortality`

A classificação foi definida da seguinte forma:

* 0 → Baixa mortalidade
* 1 → Alta mortalidade

A separação foi realizada utilizando a mediana da coluna `Total`.

Além disso, a coluna `Total` foi removida das features para evitar **Data Leakage**, garantindo que o modelo não tivesse acesso direto à variável utilizada na construção da classe alvo.

Os dados foram divididos em:

* 70% para treino
* 30% para teste

A divisão foi realizada utilizando estratificação das classes.

---

# 3. Metodologia

O trabalho foi desenvolvido utilizando Árvores de Decisão da biblioteca Scikit-Learn.

Foram realizados experimentos variando os seguintes hiperparâmetros:

| Parâmetro         | Valores Testados |
| ----------------- | ---------------- |
| max_depth         | 3, 5, 10, None   |
| criterion         | gini, entropy    |
| min_samples_leaf  | 1, 5, 10         |
| min_samples_split | 2                |

Cada configuração foi treinada e avaliada separadamente.

As métricas utilizadas para avaliação foram:

* Acurácia
* Precisão
* Recall

Também foi utilizada:

* Matriz de confusão
* Relatório de classificação
* Análise de importância das variáveis

---

# 4. Análise Descritiva

A análise estatística demonstrou forte concentração dos valores próximos de zero em diversas colunas, indicando que muitas categorias possuem baixa incidência de mortalidade.

As doenças com maior número de óbitos foram:

| Categoria                                     | Total de Óbitos |
| --------------------------------------------- | --------------- |
| Infarto agudo do miocárdio                    | 942.480         |
| Pneumonia por microorganismo não especificado | 769.500         |
| Doenças pulmonares obstrutivas crônicas       | 497.870         |

A análise também mostrou elevada variabilidade nos dados, evidenciada pelos altos valores de desvio padrão.

---

# 5. Resultados

Os modelos apresentaram desempenhos distintos conforme os hiperparâmetros utilizados.

O melhor desempenho foi obtido utilizando:

* `max_depth = None`
* `criterion = gini`

Resultados do melhor modelo:

| Métrica  | Valor  |
| -------- | ------ |
| Acurácia | 0.9904 |
| Precisão | 0.9903 |
| Recall   | 0.9903 |

Os resultados demonstram que árvores mais profundas conseguiram capturar melhor os padrões presentes nos dados.

---

# 6. Importância das Variáveis

A análise de importância das features mostrou que os capítulos CID mais relevantes para a classificação foram:

| Feature | Importância |
| ------- | ----------- |
| Cap XX  | 0.579       |
| Cap II  | 0.162       |
| Cap IX  | 0.138       |
| Cap XI  | 0.119       |

Esses resultados indicam que causas externas, neoplasias e doenças circulatórias possuem grande influência na previsão de alta mortalidade.

---

# 7. Conclusão

O trabalho demonstrou a aplicação prática de Árvores de Decisão em um problema real de análise de mortalidade utilizando dados públicos do DATASUS.

Foi possível observar que:

* O pré-processamento dos dados foi essencial para o funcionamento correto do modelo;
* A remoção de Data Leakage melhorou a confiabilidade dos resultados;
* Árvores mais profundas apresentaram melhor desempenho;
* Algumas variáveis possuem influência significativamente maior na classificação.

Além disso, o trabalho permitiu compreender melhor:

* O comportamento dos hiperparâmetros;
* O impacto das métricas de avaliação;
* A importância da análise descritiva antes da modelagem.

Os resultados obtidos demonstram que Árvores de Decisão são modelos eficientes e interpretáveis para tarefas de classificação em saúde pública.

---

# 8. Referências

* Scikit-Learn Documentation
* Pandas Documentation
* DATASUS — Sistema de Informações sobre Mortalidade (SIM)
* UCI Machine Learning Repository
* Python Software Foundation
