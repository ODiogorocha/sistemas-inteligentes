import sys
import os
import pandas as pd
import time

# --- Configuração de Caminhos ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(current_dir) == "src":
    BASE_DIR = os.path.dirname(current_dir)
    sys.path.insert(0, current_dir)
else:
    BASE_DIR = current_dir
    sys.path.insert(0, os.path.join(current_dir, "src"))

DATA_DIR = os.path.join(BASE_DIR, "data")
ORIGINAL_PATH = os.path.join(DATA_DIR, "cbr_psychology_110_cases_clinical.csv")
RUNTIME_PATH = os.path.join(DATA_DIR, "cases_runtime.csv")

try:
    from retrieval import retrieve_cases
    from adaptation import adapt_solution
    from main import build_casebase, compute_ranges
except ImportError as e:
    print(f"❌ Erro ao importar módulos: {e}")
    sys.exit(1)

def print_header(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def test_visual_adicao_remocao():
    print_header("TESTE 1: MANIPULAÇÃO DINÂMICA DO CSV")
    df = pd.read_csv(ORIGINAL_PATH)
    
    print(f"1. Lendo base original... Encontrados {len(df)} casos.")
    time.sleep(1)
    
    print("2. Criando um NOVO CASO para adicionar...")
    novo_id = "TEST_999"
    novo_caso = df.iloc[0].copy()
    novo_caso["case_id"] = novo_id
    novo_caso["main_issue"] = "Paciente com sintomas agudos de ansiedade pós-pandemia."
    
    # Adição
    df_novo = pd.concat([df, pd.DataFrame([novo_caso])], ignore_index=True)
    df_novo.to_csv(RUNTIME_PATH, index=False)
    print(f"   [OK] Caso {novo_id} salvo em 'cases_runtime.csv'.")
    print(f"   [INFO] Nova contagem: {len(df_novo)} casos.")
    time.sleep(1.5)
    
    # Remoção
    print(f"3. Removendo o caso {novo_id} da base de tempo de execução...")
    df_pos_remocao = df_novo[df_novo["case_id"] != novo_id]
    df_pos_remocao.to_csv(RUNTIME_PATH, index=False)
    print(f"   [OK] Caso removido com sucesso.")
    print(f"   [INFO] Contagem final: {len(df_pos_remocao)} casos.")

def test_visual_cbr_inédito():
    print_header("TESTE 2: CICLO CBR PARA CASO SEM SOLUÇÃO")
    df_orig = pd.read_csv(ORIGINAL_PATH)
    casebase = build_casebase(df_orig)
    ranges = compute_ranges(df_orig)
    
    print("1. DEFININDO O PROBLEMA (Query inédita):")
    query = {
        "age": 25, "clinical_severity": "severe", 
        "main_issue": "Crises de pânico recorrentes e insônia severa após evento traumático.",
        "trauma_history": "yes", "panic_symptoms": "yes", "gad7_estimate": 19
    }
    print(f"   - Problema: {query['main_issue']}")
    print(f"   - Severidade: {query['clinical_severity']}")
    time.sleep(1.5)
    
    print("\n2. RECUPERAÇÃO (Retrieve): Buscando os casos mais similares...")
    retrieved = retrieve_cases(casebase, query, k=3, ranges=ranges)
    for i, (sim, case) in enumerate(retrieved, 1):
        print(f"   [{i}] Similaridade: {sim:.4f} | Caso ID: {case['problem'].get('case_id')} | Resumo: {case['problem']['main_issue'][:40]}...")
    time.sleep(1.5)
    
    print("\n3. ADAPTAÇÃO (Adapt): Gerando recomendação personalizada...")
    adapted = adapt_solution(retrieved, query)
    
    print("\n" + "-"*40)
    print("   RESULTADO DA ADAPTAÇÃO:")
    print(f"   > Intervenção: {adapted['intervention_type'].upper()}")
    print(f"   > Intensidade: {adapted['intensity']} (ajustada pela severidade)")
    print(f"   > Frequência:  {adapted['weekly_frequency']} sessões/semana")
    print(f"   > Recomendação Gerada:")
    print(f"     {adapted['recommendation_text']}")
    print("-"*40)

if __name__ == "__main__":
    try:
        test_visual_adicao_remocao()
        test_visual_cbr_inédito()
        print("\n✅ Todos os testes visuais foram concluídos!")
    except Exception as e:
        print(f"\n❌ Ocorreu um erro durante a execução: {e}")
