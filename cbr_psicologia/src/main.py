import os
import sys
import pandas as pd

# =====================================================
# PATH
# =====================================================
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retrieval import retrieve_cases
from adaptation import adapt_solution
from evaluation import evaluate
from validation import (
    leave_one_out,
    kfold_cross_validation
)

# =====================================================
# PATHS
# =====================================================
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(BASE_DIR, "data")

ORIGINAL_PATH = os.path.join(
    DATA_DIR,
    "cbr_psychology_110_cases_clinical.csv"
)

# =====================================================
# COLUNAS
# =====================================================
SOLUTION_KEYS = [
    "intervention_type",
    "intensity",
    "weekly_frequency",
    "recommendation_text"
]

# =====================================================
# LOAD DATA
# =====================================================
def load_data():

    df = pd.read_csv(ORIGINAL_PATH)

    df["clinical_severity"] = df["clinical_severity"].replace({
        "high": "severe"
    })

    if "case_id" not in df.columns:
        df.insert(
            0,
            "case_id",
            [f"C{i:03d}" for i in range(1, len(df) + 1)]
        )

    return df


# =====================================================
# CASEBASE
# =====================================================
def build_casebase(df):

    return [
        {
            "problem": {
                k: v for k, v in row.to_dict().items()
                if k not in SOLUTION_KEYS
            },
            "solution": {
                k: v for k, v in row.to_dict().items()
                if k in SOLUTION_KEYS
            }
        }
        for _, row in df.iterrows()
    ]


# =====================================================
# RANGES
# =====================================================
def compute_ranges(df):

    return {
        col: {"min": float(df[col].min()), "max": float(df[col].max())}
        for col in df.select_dtypes(include=["int64", "float64"]).columns
    }


# =====================================================
# LISTA
# =====================================================
def list_cases(df):

    print("\n=== LISTA DE CASOS ===")

    cols = ["case_id", "main_issue", "clinical_severity"]

    temp = df[cols].copy()

    temp["main_issue"] = temp["main_issue"].astype(str).str.slice(0, 50) + "..."

    print(temp.to_string(index=False))


# =====================================================
# SUMMARY
# =====================================================
def show_dataset_summary(df):

    print("\n=== DATASET ===")
    print(f"Total de casos: {len(df)}")

    print("\nSeveridade:")
    print(df["clinical_severity"].value_counts().to_string())


# =====================================================
# OPÇÃO 4 - CBR COMPLETO (SUGESTÃO + DICA)
# =====================================================
def test_existing_case(df, casebase, ranges):

    list_cases(df)

    case_id = input("\nDigite o ID: ").strip()

    selected = df[df["case_id"].astype(str) == case_id]

    if selected.empty:
        print("Caso nao encontrado.")
        return

    row = selected.iloc[0].to_dict()

    problem = {
        k: v for k, v in row.items()
        if k not in SOLUTION_KEYS
    }

    temp_base = [
        c for c in casebase
        if str(c["problem"].get("case_id")) != case_id
    ]

    retrieved = retrieve_cases(
        temp_base,
        problem,
        5,
        ranges
    )

    pred = adapt_solution(retrieved, problem)

    suggestion = pred["clinical_severity"]
    confidence = retrieved[0][0] if retrieved else 0.0
    advice = pred["recommendation"]

    # =====================================================
    # SUGESTÃO CBR
    # =====================================================
    print("\n=== SUGESTÃO CBR ===")
    print(f"Severidade sugerida: {suggestion}")
    print(f"Confiança: {confidence:.4f}")

    # =====================================================
    # RESULTADO CBR
    # =====================================================
    print("\n=== RESULTADO CBR ===")
    print(f"Severidade real: {row['clinical_severity']}")
    print(f"Severidade predita: {suggestion}")
    print(f"Confiança: {confidence:.4f}")

    # =====================================================
    # DICA DE SOLUÇÃO (NOVO)
    # =====================================================
    print("\n=== DICA DE SOLUÇÃO ===")
    print(advice)

    # =====================================================
    # VIZINHOS
    # =====================================================
    print("\n=== VIZINHOS ===")

    for sim, case in retrieved:

        print(
            f"Sim={sim:.4f} | "
            f"ID={case['problem']['case_id']} | "
            f"Severity={case['problem']['clinical_severity']}"
        )


# =====================================================
# NOVO CASO
# =====================================================
def run_new_case(casebase, ranges):

    print("\n=== NOVO CASO ===")

    query = {}

    for col in ranges.keys():
        try:
            query[col] = float(input(f"{col}: "))
        except:
            query[col] = 0.0

    query["main_issue"] = input("main_issue: ")

    retrieved = retrieve_cases(casebase, query, 5, ranges)

    pred = adapt_solution(retrieved, query)

    print("\n=== PREDIÇÃO ===")
    print("Severidade:", pred["clinical_severity"])
    print("Dica:", pred["recommendation"])


# =====================================================
# MAIN
# =====================================================
def main():

    df = load_data()
    casebase = build_casebase(df)
    ranges = compute_ranges(df)

    while True:

        print("\n==========================================")
        print("    SISTEMA CBR - PSICOLOGIA CLINICA")
        print("==========================================")

        print("  1 - Avaliacao Leave-One-Out")
        print("  2 - Avaliacao K-Fold")
        print("  3 - Inserir novo caso")
        print("  4 - Testar caso existente")
        print("  5 - Listar casos")
        print("  6 - Analise descritiva")
        print("  0 - Sair")

        print("------------------------------------------")

        op = input("Escolha: ").strip()

        if op == "1":

            y_t, y_p, _, _, _, _ = leave_one_out(casebase, ranges)
            metrics = evaluate(y_t, y_p)

            print("\n=== RESULTADOS ===")
            print(metrics)

        elif op == "2":

            y_t, y_p, _, _, _, _ = kfold_cross_validation(casebase, ranges)
            metrics = evaluate(y_t, y_p)

            print("\n=== RESULTADOS ===")
            print(metrics)

        elif op == "3":
            run_new_case(casebase, ranges)

        elif op == "4":
            test_existing_case(df, casebase, ranges)

        elif op == "5":
            list_cases(df)

        elif op == "6":
            show_dataset_summary(df)

        elif op == "0":
            break

        else:
            print("Opcao invalida.")


if __name__ == "__main__":
    main()