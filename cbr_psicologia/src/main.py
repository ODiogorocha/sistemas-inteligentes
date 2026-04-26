import os
import sys
import pandas as pd

# Garante que o Python encontre os módulos locais
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retrieval  import retrieve_cases
from adaptation import adapt_solution
from evaluation import evaluate, adaptation_error
from validation import leave_one_out, kfold_cross_validation

# ------------------------------------------------------------------
# Caminhos
# ------------------------------------------------------------------
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR      = os.path.join(BASE_DIR, "data")
ORIGINAL_PATH = os.path.join(DATA_DIR, "cbr_psychology_110_cases_clinical.csv")
RUNTIME_PATH  = os.path.join(DATA_DIR, "cases_runtime.csv")

SOLUTION_KEYS = ["intervention_type", "intensity", "weekly_frequency", "recommendation_text"]

NUMERIC_FEATURES = [
    "age", "anxiety_score", "depression_score", "stress_level",
    "sleep_quality", "sleep_hours", "symptom_duration_months",
    "gad7_estimate", "phq9_estimate", "irritability_level",
    "bmi_estimate",
]

CATEGORICAL_FEATURES = [
    "gender", "social_support", "physical_activity", "panic_symptoms",
    "concentration_difficulty", "appetite_change", "prior_treatment",
    "current_medication", "trauma_history", "substance_use_risk",
    "work_or_study_impairment", "comorbid_profile", "clinical_severity",
]

# ------------------------------------------------------------------
# Carga e persistência
# ------------------------------------------------------------------
def load_data():
    df_orig = pd.read_csv(ORIGINAL_PATH)
    if "case_id" not in df_orig.columns:
        df_orig.insert(0, "case_id", [f"C{i:03d}" for i in range(1, len(df_orig) + 1)])

    if os.path.exists(RUNTIME_PATH):
        df_rt = pd.read_csv(RUNTIME_PATH)
        if df_rt.empty:
            df_rt = df_orig.copy()
            df_rt.to_csv(RUNTIME_PATH, index=False)
    else:
        df_rt = df_orig.copy()
        df_rt.to_csv(RUNTIME_PATH, index=False)

    return df_orig, df_rt

def save_data(df):
    df.to_csv(RUNTIME_PATH, index=False)

def build_casebase(df):
    casebase = []
    for _, row in df.iterrows():
        data = row.to_dict()
        problem = {k: v for k, v in data.items() if k not in SOLUTION_KEYS}
        solution = {k: v for k, v in data.items() if k in SOLUTION_KEYS}
        casebase.append({"problem": problem, "solution": solution})
    return casebase

def compute_ranges(df):
    ranges = {}
    for col in df.select_dtypes(include=["int64", "float64"]).columns:
        ranges[col] = {"min": float(df[col].min()), "max": float(df[col].max())}
    return ranges

# ------------------------------------------------------------------
# Funções de Menu
# ------------------------------------------------------------------
def list_cases(df):
    print("\n=== LISTA DE CASOS ===")
    if df.empty:
        print("Aviso: A base de dados atual esta vazia.")
    else:
        cols = ["case_id", "main_issue", "clinical_severity", "intervention_type"]
        temp_df = df[cols].copy()
        temp_df["main_issue"] = temp_df["main_issue"].str.slice(0, 50) + "..."
        print(temp_df.to_string(index=False))

def show_dataset_summary(df):
    print("\n=== ANALISE DESCRITIVA DO DATASET ===")
    print(f"Total de casos: {len(df)}")
    print("\nDistribuicao por Severidade:")
    print(df["clinical_severity"].value_counts().to_string())
    print("\nTipos de Intervencao:")
    print(df["intervention_type"].value_counts().to_string())
    print("\nEstatisticas de Scores (Media):")
    cols = ["anxiety_score", "depression_score", "stress_level", "sleep_quality"]
    print(df[cols].mean().round(2).to_string())

def test_existing_case(df, casebase, ranges):
    list_cases(df)
    case_id = input("\nDigite o ID do caso para testar: ").strip()
    selected = df[df["case_id"].astype(str) == case_id]
    
    if selected.empty:
        print("Erro: Caso nao encontrado.")
        return

    row = selected.iloc[0].to_dict()
    problem = {k: v for k, v in row.items() if k not in SOLUTION_KEYS}
    
    print(f"\nTestando Caso {case_id}...")
    # Remove o proprio caso da base temporariamente para um teste real
    temp_base = [c for c in casebase if str(c["problem"].get("case_id")) != case_id]
    
    retrieved = retrieve_cases(temp_base, problem, 3, ranges)
    print("\n--- Casos Similares Encontrados ---")
    for sim, case in retrieved:
        print(f"  Sim: {sim:.4f} | ID: {case['problem'].get('case_id')} | Tipo: {case['solution']['intervention_type']}")
    
    adapted = adapt_solution(retrieved, problem)
    print("\n--- Solucao Adaptada vs Real ---")
    print(f"  Adaptada: {adapted['intervention_type']} (Intensidade: {adapted['intensity']})")
    print(f"  Real:     {row['intervention_type']} (Intensidade: {row['intensity']})")

def add_new_case_to_csv(df):
    print("\n=== ADICIONAR NOVO CASO AO CSV ===")
    novo_caso = {}
    last_id = str(df["case_id"].iloc[-1])
    try:
        if last_id.startswith("C"):
            new_id = f"C{int(last_id[1:]) + 1:03d}"
        else:
            new_id = int(last_id) + 1
    except:
        new_id = "N999"
    
    novo_caso["case_id"] = new_id
    for feat in NUMERIC_FEATURES:
        while True:
            val = input(f"  {feat}: ").strip()
            try:
                novo_caso[feat] = float(val)
                break
            except:
                print("  Valor invalido.")
                
    for feat in CATEGORICAL_FEATURES:
        novo_caso[feat] = input(f"  {feat}: ").strip()
    
    novo_caso["main_issue"] = input("  main_issue: ").strip()
    novo_caso["intervention_type"] = input("  intervention_type: ").strip() or "psychotherapy"
    novo_caso["intensity"] = float(input("  intensity (1-5): ").strip() or 3.0)
    novo_caso["weekly_frequency"] = float(input("  weekly_frequency: ").strip() or 2.0)
    novo_caso["recommendation_text"] = input("  recommendation_text: ").strip() or "Avaliacao."

    df_novo = pd.concat([df, pd.DataFrame([novo_caso])], ignore_index=True)
    save_data(df_novo)
    print(f"Sucesso: Caso {new_id} adicionado.")
    return df_novo

def run_new_case(casebase, ranges):
    print("\n=== CONSULTA DE NOVO CASO ===")
    query = {}
    for feat in ["age", "anxiety_score", "depression_score"]:
        query[feat] = float(input(f"  {feat}: ") or 0)
    query["clinical_severity"] = input("  clinical_severity (mild/moderate/severe): ")
    query["main_issue"] = input("  main_issue: ")
    
    retrieved = retrieve_cases(casebase, query, 3, ranges)
    adapted = adapt_solution(retrieved, query)
    print(f"\nRecomendacao: {adapted['intervention_type']} | {adapted['recommendation_text'][:100]}...")

def main():
    df_original, df = load_data()

    while True:
        print("\n" + "=" * 42)
        print("    SISTEMA CBR - PSICOLOGIA CLINICA")
        print("=" * 42)
        print("  1 - Avaliacao Leave-One-Out")
        print("  2 - Avaliacao K-Fold")
        print("  3 - Inserir novo caso (inferencia)")
        print("  4 - Testar caso existente")
        print("  5 - Remover caso da base (CSV)")
        print("  6 - Listar casos (Resumo)")
        print("  7 - Ver todos os casos (Completo)")
        print("  8 - Analise descritiva")
        print("  9 - Adicionar novo caso ao CSV")
        print("  0 - Sair")
        print("-" * 42)
        
        op = input("Escolha: ").strip()
        casebase = build_casebase(df)
        ranges   = compute_ranges(df)

        if   op == "1": 
            y_t, y_p, _, _, _, _ = leave_one_out(casebase, ranges)
            print("\nAcuracia:", evaluate(y_t, y_p)["accuracy"])
        elif op == "2": 
            y_t, y_p, _, _, _, _ = kfold_cross_validation(casebase, ranges)
            print("\nAcuracia:", evaluate(y_t, y_p)["accuracy"])
        elif op == "3": run_new_case(casebase, ranges)
        elif op == "4": test_existing_case(df, casebase, ranges)
        elif op == "5": df = remove_case(df)
        elif op == "6": list_cases(df)
        elif op == "7": print(df.to_string())
        elif op == "8": show_dataset_summary(df)
        elif op == "9": df = add_new_case_to_csv(df)
        elif op == "0": break
        else: print("Opcao invalida.")

if __name__ == "__main__":
    main()
