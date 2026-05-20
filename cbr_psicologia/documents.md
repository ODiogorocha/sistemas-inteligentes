# Explicação Técnica Completa do Sistema

---

# main.py

O arquivo `main.py` é o núcleo do sistema.

Ele controla:

- carregamento dos dados;
- menu principal;
- chamadas do CBR;
- validações;
- interação com usuário.

---

# Importações

```python
import os
import sys
import pandas as pd
```

## Objetivo

### os

Manipulação de caminhos e arquivos.

### sys

Permite alterar o PATH do Python.

### pandas

Leitura e manipulação do dataset CSV.

---

# Ajuste do PATH

```python
sys.path.insert(
    0,
    os.path.dirname(
        os.path.abspath(__file__)
    )
)
```

## Objetivo

Permitir que os módulos:

- retrieval.py
- similarity.py
- adaptation.py

possam ser importados corretamente.

---

# Importação dos módulos internos

```python
from retrieval import retrieve_cases
from adaptation import adapt_solution
from evaluation import evaluate
```

## Objetivo

Separar responsabilidades do sistema.

---

# Vantagem da modularização

Cada arquivo possui função específica.

| Arquivo | Responsabilidade |
|---|---|
| similarity.py | Similaridade |
| retrieval.py | Recuperação |
| adaptation.py | Reutilização |
| validation.py | Validação |
| evaluation.py | Métricas |

---

# Caminhos do Dataset

```python
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)
```

## Objetivo

Encontrar automaticamente a pasta raiz do projeto.

---

# Caminho do CSV

```python
ORIGINAL_PATH = os.path.join(
    DATA_DIR,
    "cbr_psychology_110_cases_clinical.csv"
)
```

## Objetivo

Localizar o dataset clínico.

---

# load_data()

Responsável por:

- carregar CSV;
- normalizar dados;
- preparar dataset.

---

# Leitura do CSV

```python
df_orig = pd.read_csv(
    ORIGINAL_PATH
)
```

## Objetivo

Carregar os 110 casos clínicos.

---

# Normalização da Severidade

```python
df_orig["clinical_severity"] = (
    df_orig["clinical_severity"]
    .replace({
        "high": "severe"
    })
)
```

## Por que isso foi feito?

Padronização das classes.

Sem isso:

```txt
high != severe
```

Isso prejudicaria a similaridade.

---

# Redução de Classes

```python
CLASS_MAPPING = {
    "therapy": "therapy",
    "combined": "therapy",
    "mindfulness": "support",
}
```

## Objetivo

Reduzir excesso de classes terapêuticas.

---

# Justificativa

Classes demais:

- aumentam ruído;
- reduzem generalização;
- diminuem acurácia.

---

# Criação do Case ID

```python
df_orig.insert(
    0,
    "case_id",
    [
        f"C{i:03d}"
    ]
)
```

## Objetivo

Criar identificador único.

Exemplo:

```txt
C001
C002
C003
```

---

# build_casebase()

Transforma DataFrame em estrutura CBR.

---

# Estrutura de Caso

```python
{
    "problem": problem,
    "solution": solution
}
```

## Objetivo

Separar:

| Parte | Conteúdo |
|---|---|
| problem | sintomas |
| solution | tratamento |

---

# Compatibilidade com CBRKit

Essa estrutura segue diretamente o padrão do CBRKit.

---

# compute_ranges()

Calcula:

- mínimo;
- máximo;

de atributos numéricos.

---

# Objetivo

Normalizar similaridade numérica.

---

# Exemplo

Sem normalização:

```txt
idade = 80
ansiedade = 8
```

A idade dominaria a distância.

---

# similarity.py

Responsável pelo cálculo de similaridade.

---

# FEATURE_WEIGHTS

```python
FEATURE_WEIGHTS = {
    "gad7_estimate": 5.0,
    "phq9_estimate": 5.0,
}
```

## Objetivo

Dar importância diferente para atributos clínicos.

---

# Justificativa Clínica

## GAD-7

Escala validada para ansiedade.

## PHQ-9

Escala validada para depressão.

---

# Variáveis ignoradas

```python
"gender": 0.0,
```

## Motivo

Evitar:

- viés;
- ruído;
- atributos irrelevantes.

---

# numeric_similarity()

```python
sim = 1.0 - (dist / max_dist)
```

---

# Explicação Matemática

## dist

Distância entre valores.

## max_dist

Maior distância possível.

---

# Resultado

| Valores | Similaridade |
|---|---|
| 8 vs 9 | Alta |
| 2 vs 9 | Baixa |

---

# categorical_similarity()

Compara atributos categóricos.

---

# Exemplo

```python
"moderate" vs "severe"
```

---

# Matrizes Clínicas

```python
SEVERITY_MATRIX
```

---

# Objetivo

Permitir similaridade parcial.

---

# Exemplo

```python
("moderate", "severe"): 0.5
```

---

# Justificativa Clínica

Casos moderados podem possuir proximidade clínica com severos.

---

# retrieval.py

Responsável pela recuperação dos casos.

---

# retrieve_cases()

Executa:

1. cálculo de similaridade;
2. ranking;
3. recuperação.

---

# Fluxo

```txt
Caso → Similaridade → Ordenação → Recuperação
```

---

# Ordenação

```python
results.sort(
    key=lambda x: x[0],
    reverse=True
)
```

## Objetivo

Maior similaridade primeiro.

---

# Filtro Clínico

```python
if r[0] >= 0.65
```

## Objetivo

Eliminar vizinhos pouco relevantes.

---

# adaptation.py

Responsável pelo Reuse.

---

# adapt_solution()

Reutiliza soluções anteriores.

---

# Majority Voting

```python
Counter(severities).most_common(1)
```

---

# Objetivo

Selecionar severidade dominante.

---

# Vantagens

- estabilidade;
- robustez;
- redução de ruído.

---

# Recomendação Clínica

O sistema reutiliza:

```python
recommendation_text
```

dos vizinhos.

---

# Isso é extremamente importante

Porque caracteriza o verdadeiro:

```txt
Reuse do CBR
```

---

# validation.py

Responsável pela validação estatística.

---

# Leave-One-Out

## Funcionamento

1. Remove 1 caso.
2. Treina nos restantes.
3. Prediz o removido.

---

# Justificativa

Excelente para datasets pequenos.

---

# K-Fold

## Funcionamento

1. Divide em K partes.
2. Testa uma parte.
3. Treina no restante.

---

# Vantagem

Menor variância estatística.

---

# evaluation.py

Responsável pelas métricas.

---

# Accuracy

Percentual total de acertos.

---

# Precision

Qualidade das previsões positivas.

---

# Recall

Capacidade de encontrar casos corretos.

---

# F1-score

Equilíbrio entre precision e recall.

---

# Fluxo Completo do Sistema

## Etapa 1

Usuário escolhe:

```txt
C110
```

---

## Etapa 2

Sistema monta:

```python
problem = {...}
```

---

## Etapa 3

Retrieval busca casos semelhantes.

---

## Etapa 4

Similarity calcula distância ponderada.

---

## Etapa 5

Adaptation reutiliza soluções.

---

## Etapa 6

Sistema exibe:

- severidade;
- confiança;
- vizinhos;
- recomendação.

---

# Explicabilidade

O sistema mostra:

```txt
Sim=0.8252 | ID=C005
```

---

# Importância

Sistemas clínicos precisam ser:

- interpretáveis;
- auditáveis;
- explicáveis.

---

# Diferença para Machine Learning Tradicional

## ML tradicional

Funciona como caixa preta.

---

## CBR

Mostra:

- quais casos influenciaram;
- por que decidiu;
- confiança.

---

# Compatibilidade com CBRKit

O projeto segue:

| Conceito | Implementação |
|---|---|
| Case Representation | problem/solution |
| Retrieval | retrieve_cases |
| Reuse | adapt_solution |
| Similarity | similarity.py |
| Validation | validation.py |

---

# Pontos Fortes do Projeto

## Interpretabilidade

Explica decisões.

---

## Modularização

Arquitetura limpa.

---

## Escalabilidade

Novos casos podem ser adicionados.

---

## Robustez

Pesos clínicos reduzem ruído.

---

## Fundamentação Científica

Utiliza:

- similarity weighting;
- retrieval;
- reuse;
- validação;
- métricas estatísticas.

---

# Possíveis Perguntas do Professor

---

## “Por que usar CBR?”

Porque psicologia clínica depende fortemente de experiências anteriores e análise comparativa de sintomas.

---

## “Por que usar pesos?”

Porque diferentes variáveis possuem relevância clínica diferente.

---

## “Por que usar matrizes?”

Porque categorias clínicas possuem proximidade parcial.

---

## “Por que usar Leave-One-Out?”

Porque o dataset é pequeno e o método aproveita melhor os dados disponíveis.

---

## “Qual a principal vantagem?”

Explicabilidade clínica baseada em casos reais semelhantes.

---

# Conclusão

O sistema implementa um CBR completo para psicologia clínica.

Ele:

- recupera casos;
- calcula similaridade;
- reutiliza soluções;
- gera recomendações;
- explica decisões;
- valida resultados estatisticamente.

Além disso, segue arquitetura compatível com o paradigma do CBRKit.

# Explicação Linha por Linha — similarity.py

---

# Objetivo do Arquivo

O arquivo `similarity.py` é responsável por calcular o quanto dois casos clínicos são parecidos.

Esse é o núcleo matemático do sistema CBR.

Sem similaridade:

```txt
não existe Retrieval
```

---

# Estrutura Geral

O arquivo possui:

1. pesos clínicos;
2. matrizes clínicas;
3. similaridade numérica;
4. similaridade categórica;
5. cálculo global ponderado.

---

# FEATURE_WEIGHTS

```python
FEATURE_WEIGHTS = {
```

## Objetivo

Define a importância de cada atributo clínico.

---

# Por que isso é necessário?

Nem todas as variáveis possuem o mesmo impacto clínico.

Exemplo:

| Variável | Importância |
|---|---|
| GAD-7 | Alta |
| BMI | Baixa |

---

# Exemplo

```python
"gad7_estimate": 5.0,
```

---

# Interpretação

O score GAD-7 terá peso 5 vezes maior do que uma variável com peso 1.

---

# Justificativa Clínica

## GAD-7

Escala validada internacionalmente para ansiedade.

## PHQ-9

Escala validada para depressão.

Esses scores possuem alta relevância clínica.

---

# Variáveis Ignoradas

```python
"gender": 0.0,
```

---

# Objetivo

Eliminar atributos considerados ruído.

---

# Justificativa

Gênero não necessariamente determina severidade clínica.

Logo:

```txt
peso = 0
```

---

# Benefícios

- reduz viés;
- reduz overfitting;
- melhora generalização;
- melhora retrieval.

---

# SEVERITY_MATRIX

```python
SEVERITY_MATRIX = {
```

---

# Objetivo

Criar proximidade parcial entre classes clínicas.

---

# Problema sem matriz

Sem matriz:

```python
moderate != severe
```

Resultado:

```txt
similaridade = 0
```

---

# Problema Clínico

Na prática:

```txt
moderate pode estar próximo de severe
```

---

# Solução

```python
("moderate", "severe"): 0.5
```

---

# Interpretação

Existe:

- semelhança parcial;
- proximidade clínica;
- continuidade entre classes.

---

# IMPAIRMENT_MATRIX

Representa proximidade entre níveis de comprometimento funcional.

---

# Exemplo

```python
("moderate", "high"): 0.5
```

---

# Justificativa

Comprometimento moderado pode possuir características semelhantes ao alto comprometimento.

---

# numeric_similarity()

```python
def numeric_similarity(
```

---

# Objetivo

Calcular similaridade entre atributos numéricos.

---

# Exemplo

```txt
ansiedade = 8
ansiedade = 9
```

Alta similaridade.

---

# Conversão para float

```python
a = float(a)
b = float(b)
```

---

# Objetivo

Garantir operação matemática segura.

---

# Tratamento de erro

```python
except:
    return 0.0
```

---

# Objetivo

Evitar quebra do sistema.

---

# Distância

```python
dist = abs(a - b)
```

---

# Objetivo

Medir diferença absoluta entre valores.

---

# Distância máxima

```python
max_dist = max_val - min_val
```

---

# Objetivo

Normalizar os dados.

---

# Fórmula principal

```python
sim = 1.0 - (dist / max_dist)
```

---

# Explicação Matemática

## Quando os valores são iguais

```txt
dist = 0
```

Logo:

```txt
sim = 1
```

---

# Quando são muito diferentes

```txt
dist ≈ max_dist
```

Logo:

```txt
sim ≈ 0
```

---

# clamp()

```python
return max(
    0.0,
    min(1.0, sim)
)
```

---

# Objetivo

Garantir:

```txt
0 ≤ similaridade ≤ 1
```

---

# categorical_similarity()

```python
def categorical_similarity(
```

---

# Objetivo

Comparar atributos categóricos.

---

# Exemplo

```txt
moderate vs severe
```

---

# Padronização

```python
a = str(a).strip().lower()
```

---

# Objetivo

Evitar erros causados por:

- maiúsculas;
- espaços;
- inconsistências.

---

# Uso de matriz

```python
if matrix:
```

---

# Objetivo

Permitir similaridade parcial.

---

# Sem matriz

```python
return 1.0 if a == b else 0.0
```

---

# Interpretação

| Comparação | Resultado |
|---|---|
| equal | 1 |
| diferente | 0 |

---

# _select_sim()

```python
def _select_sim(
```

---

# Objetivo

Escolher automaticamente qual similaridade usar.

---

# Fluxo

| Tipo | Método |
|---|---|
| Numérico | numeric_similarity |
| Categórico | categorical_similarity |
| Severidade | matriz clínica |

---

# Verificação do peso

```python
if weight <= 0:
```

---

# Objetivo

Ignorar atributos irrelevantes.

---

# Similaridade Numérica

```python
if isinstance(v1, (int, float)):
```

---

# Objetivo

Detectar atributos numéricos automaticamente.

---

# Similaridade de Severidade

```python
if key == "clinical_severity":
```

---

# Objetivo

Aplicar matriz clínica especializada.

---

# Similaridade de Impairment

```python
if key == "work_or_study_impairment":
```

---

# Objetivo

Aplicar matriz específica para comprometimento funcional.

---

# compute_similarity()

```python
def compute_similarity(
```

---

# Objetivo

Calcular similaridade global entre dois casos.

---

# Variáveis principais

```python
weighted_sum = 0.0
total_weight = 0.0
```

---

# Objetivo

Implementar média ponderada.

---

# Loop principal

```python
for key, v1 in case_problem.items():
```

---

# Objetivo

Percorrer todos os atributos do caso.

---

# Ignorar case_id

```python
if key == "case_id":
```

---

# Motivo

O ID não possui significado clínico.

---

# Verificar existência

```python
if key not in query:
```

---

# Objetivo

Evitar erro de chave inexistente.

---

# Similaridade local

```python
score, weight = _select_sim(
```

---

# Objetivo

Calcular similaridade individual do atributo.

---

# Soma ponderada

```python
weighted_sum += (
    score * weight
)
```

---

# Interpretação

Atributos mais importantes influenciam mais.

---

# Soma total dos pesos

```python
total_weight += weight
```

---

# Similaridade Final

```python
return weighted_sum / total_weight
```

---

# Interpretação

Média ponderada das similaridades locais.

---

# Resultado Final

A função retorna:

```txt
valor entre 0 e 1
```

---

# Exemplo

| Similaridade | Interpretação |
|---|---|
| 0.95 | muito parecido |
| 0.80 | parecido |
| 0.50 | parcialmente parecido |
| 0.20 | pouco parecido |

---

# Importância no Sistema

Essa função é o núcleo do Retrieval.

Ela determina:

- quais casos serão recuperados;
- ranking dos vizinhos;
- qualidade da recomendação;
- acurácia do sistema.

---

# Compatibilidade com CBRKit

O CBRKit utiliza exatamente o conceito de:

```txt
Local Similarity + Global Similarity
```

---

# O sistema implementa:

| Conceito CBRKit | Implementação |
|---|---|
| Local Similarity | numeric_similarity |
| Local Similarity | categorical_similarity |
| Global Similarity | compute_similarity |
| Weighting | FEATURE_WEIGHTS |

---

# Decisões Técnicas Importantes

## Uso de pesos

Permite relevância clínica diferenciada.

---

## Uso de matrizes

Permite proximidade parcial.

---

## Normalização

Evita distorção de escala.

---

## Média ponderada

Melhora robustez do Retrieval.

---

# Conclusão

O arquivo `similarity.py` implementa o motor matemático do sistema CBR.

Ele:

- calcula similaridade;
- aplica pesos clínicos;
- utiliza matrizes clínicas;
- normaliza atributos;
- gera similaridade global ponderada.

Essa etapa é fundamental porque controla toda a qualidade do Retrieval e das recomendações clínicas.

# Explicação Linha por Linha — retrieval.py

---

# Objetivo do Arquivo

O arquivo `retrieval.py` implementa a etapa de:

```txt
Retrieval
```

do ciclo CBR.

---

# O que é Retrieval?

Retrieval é o processo de:

```txt
buscar casos semelhantes
```

na base de conhecimento.

---

# Importação

```python
from similarity import compute_similarity
```

---

# Objetivo

Importar a função responsável pelo cálculo de similaridade global.

---

# retrieve_cases()

```python
def retrieve_cases(
```

---

# Objetivo

Receber:

- base de casos;
- consulta atual;
- número de vizinhos;
- ranges numéricos;

e retornar os casos mais similares.

---

# Parâmetros

| Parâmetro | Função |
|---|---|
| casebase | base de conhecimento |
| query | novo caso |
| k | quantidade de vizinhos |
| ranges | normalização numérica |

---

# Lista de resultados

```python
results = []
```

---

# Objetivo

Armazenar:

```python
(similaridade, caso)
```

---

# Loop principal

```python
for case in casebase:
```

---

# Objetivo

Percorrer todos os casos existentes.

---

# Similaridade

```python
sim = compute_similarity(
```

---

# Objetivo

Calcular o quanto o caso atual é parecido com a consulta.

---

# Estrutura usada

```python
case["problem"]
```

---

# Motivo

No CBR:

| Parte | Conteúdo |
|---|---|
| problem | sintomas |
| solution | solução |

---

# Armazenamento

```python
results.append(
    (sim, case)
)
```

---

# Objetivo

Guardar:

- score;
- caso correspondente.

---

# Exemplo

```python
(0.82, caso_C005)
```

---

# Ordenação

```python
results.sort(
    key=lambda x: x[0],
    reverse=True
)
```

---

# Objetivo

Ordenar pelos maiores scores de similaridade.

---

# Explicação

## x[0]

Representa:

```python
sim
```

---

# reverse=True

Coloca os maiores valores primeiro.

---

# Resultado

| Similaridade | Ordem |
|---|---|
| 0.91 | 1º |
| 0.84 | 2º |
| 0.70 | 3º |

---

# Filtro Clínico

```python
filtered = [
    r
    for r in results
    if r[0] >= 0.65
]
```

---

# Objetivo

Eliminar casos pouco relevantes.

---

# Justificativa Clínica

Casos muito diferentes:

- confundem o sistema;
- reduzem acurácia;
- introduzem ruído.

---

# Por que 0.65?

Foi definido empiricamente.

---

# Interpretação

| Score | Interpretação |
|---|---|
| > 0.80 | muito similar |
| 0.65–0.80 | relevante |
| < 0.65 | pouco confiável |

---

# Seleção Final

```python
return filtered[:k]
```

---

# Objetivo

Retornar somente os k melhores vizinhos.

---

# Exemplo

Se:

```python
k = 5
```

Retorna:

```txt
5 casos mais similares
```

---

# Fallback

```python
return results[:k]
```

---

# Objetivo

Garantir funcionamento mesmo sem casos acima do threshold.

---

# Fluxo Completo do Retrieval

---

# Etapa 1

Recebe novo caso.

---

# Etapa 2

Percorre toda base.

---

# Etapa 3

Calcula similaridade.

---

# Etapa 4

Ordena resultados.

---

# Etapa 5

Filtra vizinhos relevantes.

---

# Etapa 6

Retorna top-k vizinhos.

---

# Exemplo Real

## Entrada

Caso:

```txt
C110
```

---

# Similaridades calculadas

```txt
C005 → 0.8252
C072 → 0.7637
C085 → 0.7569
```

---

# Resultado

```txt
Top 5 vizinhos mais similares
```

---

# Relação com CBRKit

O CBRKit utiliza exatamente essa lógica.

---

# Conceitos implementados

| Conceito | Implementação |
|---|---|
| Retrieval | retrieve_cases |
| Similarity Ranking | sort |
| Top-k Retrieval | [:k] |
| Threshold Filtering | >= 0.65 |

---

# Por que Retrieval é importante?

Sem Retrieval:

```txt
não existe CBR
```

---

# O Retrieval controla

- qualidade das recomendações;
- acurácia;
- interpretabilidade;
- confiança.

---

# Importância Clínica

Na psicologia clínica:

```txt
casos semelhantes → intervenções semelhantes
```

---

# Explicabilidade

O sistema mostra:

```txt
Sim=0.8252 | ID=C005
```

---

# Isso é importante porque

o usuário consegue entender:

- quais casos influenciaram;
- por que a decisão foi tomada;
- nível de confiança.

---

# Comparação com Machine Learning

## ML tradicional

```txt
caixa preta
```

---

# Retrieval em CBR

```txt
explicável e interpretável
```

---

# Complexidade Computacional

O Retrieval atual possui:

```txt
O(n)
```

---

# Significado

Percorre todos os casos da base.

---

# Para bases pequenas

Isso é totalmente aceitável.

---

# Para bases grandes

Poderiam ser usados:

- KD-Tree;
- BallTree;
- FAISS;
- Annoy.

---

# Decisões Técnicas Importantes

---

# Uso de Top-k

Evita usar casos irrelevantes.

---

# Uso de Threshold

Reduz ruído.

---

# Ordenação Descrescente

Prioriza casos mais similares.

---

# Estrutura Modular

Facilita manutenção e expansão.

---

# Conclusão

O arquivo `retrieval.py` implementa a etapa de recuperação do CBR.

Ele:

- percorre a base;
- calcula similaridade;
- ordena casos;
- filtra vizinhos;
- retorna os casos mais relevantes.

Essa etapa é responsável por conectar:

```txt
novo problema ↔ experiências anteriores
```

que é exatamente o princípio central do Case-Based Reasoning.
# Explicação Linha por Linha — adaptation.py

---

# Objetivo do Arquivo

O arquivo `adaptation.py` implementa a etapa de:

```txt
Reuse / Adaptation
```

do ciclo CBR.

---

# O que é Adaptation?

Depois que o sistema encontra casos semelhantes, ele precisa:

```txt
reutilizar soluções anteriores
```

para resolver o novo problema.

---

# No sistema clínico

A adaptação decide:

- severidade clínica;
- recomendação;
- intervenção;
- confiança.

---

# Importação

```python
from collections import Counter
```

---

# Objetivo

Importar uma estrutura para contagem automática.

---

# Por que usar Counter?

Ele facilita descobrir:

```txt
qual classe aparece mais vezes
```

---

# Exemplo

```python
["moderate", "moderate", "severe"]
```

Resultado:

```python
Counter({
    "moderate": 2,
    "severe": 1
})
```

---

# adapt_solution()

```python
def adapt_solution(
```

---

# Objetivo

Receber os vizinhos recuperados e gerar:

```txt
uma solução final
```

---

# Parâmetros

| Parâmetro | Função |
|---|---|
| retrieved_cases | casos similares |
| new_case | consulta atual |

---

# Verificação de segurança

```python
if not retrieved_cases:
```

---

# Objetivo

Evitar falhas quando nenhum vizinho for encontrado.

---

# Fallback

```python
return {
    "clinical_severity":
        "moderate"
}
```

---

# Motivo

Garantir robustez do sistema.

---

# Justificativa Clínica

Em ausência de evidências:

```txt
moderate
```

é uma escolha conservadora.

---

# Lista de severidades

```python
severities = []
```

---

# Objetivo

Armazenar as classes dos vizinhos.

---

# Loop principal

```python
for sim, case in retrieved_cases:
```

---

# Objetivo

Percorrer os vizinhos recuperados.

---

# Estrutura do Retrieval

Cada item possui:

```python
(similaridade, caso)
```

---

# Extração da severidade

```python
case["problem"][
    "clinical_severity"
]
```

---

# Objetivo

Pegar a severidade clínica do vizinho.

---

# Armazenamento

```python
severities.append(...)
```

---

# Resultado esperado

Exemplo:

```python
[
    "moderate",
    "moderate",
    "severe",
    "moderate"
]
```

---

# Majority Voting

```python
severity = Counter(
    severities
).most_common(1)[0][0]
```

---

# Objetivo

Selecionar a severidade dominante.

---

# Explicação detalhada

## Counter(severities)

Conta frequência das classes.

---

# Resultado

```python
{
    "moderate": 3,
    "severe": 1
}
```

---

# most_common(1)

Retorna a classe mais frequente.

---

# Resultado

```python
[
    ("moderate", 3)
]
```

---

# [0][0]

Extrai somente:

```python
"moderate"
```

---

# Construção da solução

```python
return {
    "clinical_severity":
        severity
}
```

---

# Objetivo

Retornar a solução final do CBR.

---

# Fluxo Completo da Adaptation

---

# Etapa 1

Recebe vizinhos similares.

---

# Etapa 2

Extrai severidades.

---

# Etapa 3

Conta frequência.

---

# Etapa 4

Seleciona maioria.

---

# Etapa 5

Retorna solução final.

---

# Exemplo Real

## Vizinhos

```txt
C005 → moderate
C072 → moderate
C085 → severe
C069 → moderate
```

---

# Frequência

| Classe | Quantidade |
|---|---|
| moderate | 3 |
| severe | 1 |

---

# Resultado Final

```txt
moderate
```

---

# Por que Majority Voting?

Porque:

- reduz ruído;
- reduz influência de outliers;
- melhora estabilidade;
- melhora generalização.

---

# Alternativas Possíveis

O sistema poderia usar:

- weighted voting;
- similarity weighting;
- fuzzy adaptation;
- redes neurais.

---

# Por que NÃO usar isso?

Porque:

- aumentaria complexidade;
- reduziria interpretabilidade;
- poderia reduzir estabilidade em dataset pequeno.

---

# Importância Clínica

Na psicologia:

```txt
casos semelhantes tendem a ter condutas semelhantes
```

---

# Explicabilidade

O sistema mostra:

```txt
quais casos influenciaram a decisão
```

---

# Isso é essencial

Em sistemas clínicos:

- médicos precisam confiar;
- decisões precisam ser auditáveis;
- explicabilidade é obrigatória.

---

# Diferença para Machine Learning

## ML tradicional

Não explica facilmente:

```txt
por que decidiu
```

---

# CBR

Mostra:

- vizinhos;
- similaridade;
- evidências clínicas.

---

# Evolução para Recomendação Clínica

O sistema também pode reutilizar:

```python
recommendation_text
```

---

# Exemplo

Se vizinhos sugerem:

- terapia cognitiva;
- mindfulness;
- apoio social;

o sistema pode combinar recomendações.

---

# Isso caracteriza o verdadeiro Reuse

No ciclo clássico do CBR:

| Etapa | Implementação |
|---|---|
| Retrieve | retrieve_cases |
| Reuse | adapt_solution |

---

# Compatibilidade com CBRKit

O CBRKit define:

```txt
Reuse = adaptação da solução recuperada
```

---

# O sistema implementa exatamente isso

A solução não é criada do zero.

Ela é:

```txt
reutilizada a partir de casos anteriores
```

---

# Decisões Técnicas Importantes

---

# Majority Voting

Escolhido por estabilidade.

---

# Fallback Clínico

Evita falhas do sistema.

---

# Estrutura Simples

Melhora interpretabilidade.

---

# Reuso de Casos

Mantém fidelidade ao paradigma CBR.

---

# Possíveis Melhorias Futuras

---

# Weighted Voting

Peso maior para vizinhos mais similares.

---

# Fuzzy Adaptation

Permitir incerteza clínica.

---

# Recomendação híbrida

Combinar múltiplas intervenções.

---

# Aprendizado incremental

Inserir novos casos automaticamente.

---

# Conclusão

O arquivo `adaptation.py` implementa a etapa de reutilização do CBR.

Ele:

- recebe vizinhos;
- extrai soluções;
- aplica votação majoritária;
- gera predição final;
- mantém interpretabilidade.

Essa etapa é fundamental porque transforma:

```txt
experiências anteriores
```

em:

```txt
novas recomendações clínicas
```

seguindo exatamente os princípios do Case-Based Reasoning e da arquitetura conceitual utilizada pelo CBRKit.
# Explicação Linha por Linha — validation.py

---

# Objetivo do Arquivo

O arquivo `validation.py` é responsável por validar estatisticamente o sistema CBR.

---

# Por que validação é importante?

Sem validação:

```txt
não existe evidência de qualidade
```

---

# O professor provavelmente avaliará:

- robustez;
- consistência;
- generalização;
- acurácia.

---

# O arquivo implementa

| Método | Objetivo |
|---|---|
| Leave-One-Out | validação exaustiva |
| K-Fold | validação estatística |

---

# Importações

```python
import random
```

---

# Objetivo

Permitir embaralhamento aleatório no K-Fold.

---

# Importações internas

```python
from retrieval import retrieve_cases
from adaptation import adapt_solution
```

---

# Objetivo

Utilizar:

- recuperação;
- adaptação;

durante os testes.

---

# leave_one_out()

```python
def leave_one_out(
```

---

# Objetivo

Executar validação Leave-One-Out.

---

# O que é Leave-One-Out?

Também chamado:

```txt
LOOCV
```

---

# Funcionamento

Para cada caso:

1. remove o caso;
2. usa os demais como treino;
3. tenta prever o removido.

---

# Exemplo

Com 110 casos:

```txt
110 execuções
```

---

# Por que isso é importante?

Maximiza uso dos dados.

---

# Especialmente útil quando

o dataset é pequeno.

---

# Estruturas de resultado

```python
y_true = []
y_pred = []
```

---

# Objetivo

Armazenar:

- valores reais;
- valores previstos.

---

# Quantidade de casos

```python
n = len(casebase)
```

---

# Objetivo

Descobrir quantos testes serão feitos.

---

# Loop principal

```python
for i in range(n):
```

---

# Objetivo

Executar um teste para cada caso.

---

# Caso de teste

```python
test = casebase[i]
```

---

# Objetivo

Selecionar o caso atual como teste.

---

# Base de treino

```python
train = (
    casebase[:i]
    +
    casebase[i + 1:]
)
```

---

# Objetivo

Remover o caso de teste da base.

---

# Isso é extremamente importante

Porque evita:

```txt
vazamento de informação
```

---

# Sem isso

o sistema encontraria:

```txt
ele mesmo
```

---

# Retrieval

```python
retrieved = retrieve_cases(
```

---

# Objetivo

Buscar casos semelhantes no conjunto de treino.

---

# Adaptação

```python
pred = adapt_solution(
```

---

# Objetivo

Gerar predição final.

---

# Armazenamento do valor real

```python
y_true.append(
```

---

# Objetivo

Guardar severidade verdadeira.

---

# Armazenamento da predição

```python
y_pred.append(
```

---

# Objetivo

Guardar severidade prevista.

---

# Verbose

```python
if verbose:
```

---

# Objetivo

Permitir depuração detalhada.

---

# Exemplo

```txt
[45/110] true=moderate pred=moderate
```

---

# Retorno

```python
return (
    y_true,
    y_pred,
)
```

---

# Objetivo

Enviar resultados para cálculo das métricas.

---

# Importância do Leave-One-Out

---

# Vantagens

| Vantagem | Explicação |
|---|---|
| Usa todos os dados | máximo aproveitamento |
| Baixo viés | treino quase completo |
| Excelente para bases pequenas | ideal para 110 casos |

---

# Desvantagem

Maior custo computacional.

---

# Porém

Com 110 casos:

```txt
isso não é problema
```

---

# kfold_cross_validation()

```python
def kfold_cross_validation(
```

---

# Objetivo

Executar validação K-Fold.

---

# O que é K-Fold?

Divide a base em:

```txt
K partes
```

---

# Exemplo

Se:

```python
k_folds = 5
```

---

# O dataset é dividido em:

```txt
5 subconjuntos
```

---

# Funcionamento

---

# Etapa 1

Seleciona 1 fold como teste.

---

# Etapa 2

Usa os outros 4 como treino.

---

# Etapa 3

Repete até todos os folds serem usados.

---

# Embaralhamento

```python
random.shuffle(shuffled)
```

---

# Objetivo

Evitar viés de ordenação.

---

# Criação dos folds

```python
folds = [
    shuffled[i::k_folds]
]
```

---

# Objetivo

Dividir os dados igualmente.

---

# Estruturas de resultado

```python
y_true = []
y_pred = []
```

---

# Objetivo

Guardar resultados globais.

---

# Loop dos folds

```python
for fold_idx in range(k_folds):
```

---

# Objetivo

Executar validação em cada fold.

---

# Fold de teste

```python
test_fold = folds[fold_idx]
```

---

# Fold de treino

```python
train_fold.extend(
    folds[j]
)
```

---

# Objetivo

Construir conjunto de treinamento.

---

# Loop dos testes

```python
for test in test_fold:
```

---

# Objetivo

Executar predição em cada caso do fold.

---

# Retrieval

```python
retrieved = retrieve_cases(
```

---

# Objetivo

Buscar vizinhos semelhantes.

---

# Adaptação

```python
pred = adapt_solution(
```

---

# Objetivo

Gerar solução final.

---

# Armazenamento

```python
y_true.append(...)
y_pred.append(...)
```

---

# Objetivo

Guardar resultados para métricas.

---

# Retorno

```python
return (
    y_true,
    y_pred,
)
```

---

# Diferença entre LOO e K-Fold

| Método | Característica |
|---|---|
| Leave-One-Out | mais rigoroso |
| K-Fold | mais rápido |
| Leave-One-Out | menor viés |
| K-Fold | menor variância |

---

# Por que usar os dois?

Porque isso fortalece cientificamente o projeto.

---

# O professor verá que

o sistema foi validado de forma séria.

---

# Relação com CBR

Validação mede:

```txt
qualidade do Retrieval + Reuse
```

---

# Se a similaridade estiver ruim

a validação detecta.

---

# Se o adaptation estiver ruim

a validação detecta.

---

# Compatibilidade com CBRKit

O CBRKit recomenda:

- validação empírica;
- testes estatísticos;
- avaliação quantitativa.

---

# O sistema implementa exatamente isso

---

# Conceitos utilizados

| Conceito | Implementação |
|---|---|
| Retrieval Testing | retrieve_cases |
| Reuse Testing | adapt_solution |
| Cross Validation | K-Fold |
| Exhaustive Validation | Leave-One-Out |

---

# Complexidade Computacional

---

# Leave-One-Out

```txt
O(n²)
```

---

# Motivo

Cada caso compara com todos os outros.

---

# K-Fold

Mais eficiente.

---

# Porém

Para 110 casos:

```txt
ambos são totalmente viáveis
```

---

# Importância Científica

Sem validação:

```txt
o sistema não possui credibilidade
```

---

# A validação prova

- generalização;
- robustez;
- consistência;
- qualidade clínica.

---

# Possíveis Perguntas do Professor

---

# “Por que usar Leave-One-Out?”

Porque a base é pequena e esse método maximiza o uso dos dados.

---

# “Por que usar K-Fold também?”

Porque reduz variância estatística e fornece validação complementar.

---

# “Por que remover o caso da base?”

Para evitar vazamento de informação.

---

# “O sistema memoriza os dados?”

Não.

Ele generaliza através de similaridade clínica.

---

# “Qual a vantagem sobre Machine Learning?”

Maior interpretabilidade e explicabilidade.

---

# Conclusão

O arquivo `validation.py` implementa a validação estatística do sistema CBR.

Ele:

- testa Retrieval;
- testa Reuse;
- mede generalização;
- evita overfitting;
- valida robustez do sistema.

Isso transforma o projeto em um sistema:

```txt
cientificamente consistente
```

e adequado para apresentação acadêmica.