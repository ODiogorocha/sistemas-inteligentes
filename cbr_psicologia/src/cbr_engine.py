"""
CBR Engine - camada formal do ciclo:
Retrieve → Reuse → Revise
"""

from retrieval import retrieve_cases
from adaptation import adapt_solution


# =====================================================
# BUILD SYSTEM
# =====================================================
def build_cbr_system(casebase, ranges, k=5):

    return {
        "casebase": casebase,
        "ranges": ranges,
        "k": k
    }


# =====================================================
# RETRIEVE
# =====================================================
def retrieve(system, query):

    return retrieve_cases(
        system["casebase"],
        query,
        system["k"],
        system["ranges"]
    )


# =====================================================
# REUSE
# =====================================================
def reuse(retrieved, query):

    return adapt_solution(retrieved, query)


# =====================================================
# CBR CYCLE COMPLETO
# =====================================================
def run_cbr(system, query):

    retrieved = retrieve(system, query)

    solution = reuse(retrieved, query)

    confidence = (
        sum(sim for sim, _ in retrieved) / len(retrieved)
        if retrieved else 0.0
    )

    result = {
        "retrieved": retrieved,
        "solution": solution,
        "confidence": confidence
    }

    # 🔥 GARANTE QUE NUNCA "SUME" A SUGESTÃO
    print("\n=== SUGESTÃO CBR ===")
    print("Severidade sugerida:", solution["clinical_severity"])
    print("Confiança:", round(confidence, 4))

    return result