"""
Módulo de recuperação: retorna os k casos mais similares (kNN).
"""
from similarity import compute_similarity


def retrieve_cases(casebase, query, k, ranges):
    """
    Recupera os k casos mais próximos da query usando kNN.

    Args:
        casebase: lista de dicts com 'problem' e 'solution'
        query:    dict com as features do novo problema
        k:        número de vizinhos a retornar
        ranges:   dict com min/max de cada feature numérica

    Returns:
        Lista de tuplas (similarity_score, case) ordenada do maior para o menor.
    """
    results = []
    for case in casebase:
        sim = compute_similarity(case["problem"], query, ranges)
        results.append((sim, case))

    results.sort(key=lambda x: x[0], reverse=True)
    return results[:k]