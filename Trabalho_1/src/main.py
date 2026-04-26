from cbrkit_system import CBRKitSystem
from case_model import Case
from data_types import DataTypeCases


def main():

    cbr = CBRKitSystem("/home/diogo/Documentos/codigos/sistemas-inteligentes/Trabalho_1/database/cbr_psychology_110_cases_clinical.csv")
    
    case_model = Case()
    type_case = DataTypeCases()

    # caso novo
    new_case = type_case.new_case

    solution, results = cbr.solve(new_case)

    print("\nRecomendação:")
    print(solution)

    print("\nCasos mais similares:")
    for case, score in results:
        print(f"- {case['case_id']} | {case['main_issue']} (score: {score:.2f})")

    print("\nExplicação:")
    for case, score in results:
        print(f"Caso {case['case_id']} foi considerado similar devido a:")
        print(f"  - Problema: {case['main_issue']}")
        print(f"  - Severidade: {case['clinical_severity']}")
        print(f"  - Score: {score:.2f}\n")


if __name__ == "__main__":
    main()