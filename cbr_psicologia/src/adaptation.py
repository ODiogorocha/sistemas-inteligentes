"""
CBR Adaptation - gera solução + dica clínica baseada nos vizinhos.
"""

from collections import Counter


# =====================================================
# ADAPTAÇÃO PRINCIPAL
# =====================================================
def adapt_solution(retrieved_cases, new_case):

    if not retrieved_cases:

        return {
            "clinical_severity": "moderate",
            "recommendation": "No similar cases found. Suggest general psychological monitoring."
        }

    # =====================================================
    # SEVERIDADE (maioria dos vizinhos)
    # =====================================================
    severities = [
        case["problem"]["clinical_severity"]
        for _, case in retrieved_cases
    ]

    severity = Counter(severities).most_common(1)[0][0]

    # =====================================================
    # SOLUÇÃO (REUSE INTELIGENTE)
    # =====================================================
    solutions = [
        case["solution"].get("recommendation_text")
        for _, case in retrieved_cases
        if case["solution"].get("recommendation_text")
    ]

    if solutions:

        # pega a solução mais comum
        recommendation = Counter(solutions).most_common(1)[0][0]

    else:

        # fallback clínico inteligente
        recommendation = build_clinical_advice(severity)

    return {
        "clinical_severity": severity,
        "recommendation": recommendation
    }


# =====================================================
# DICAS CLÍNICAS BASEADAS NA SEVERIDADE
# =====================================================
def build_clinical_advice(severity):

    if severity == "mild":

        return (
            "Low intensity intervention recommended: "
            "psychoeducation, sleep hygiene improvement, "
            "and monitoring of symptoms weekly."
        )

    elif severity == "moderate":

        return (
            "Moderate intervention recommended: "
            "cognitive behavioral therapy (CBT), "
            "structured psychotherapy sessions, "
            "and stress management techniques."
        )

    else:

        return (
            "High severity detected: "
            "urgent psychiatric evaluation recommended, "
            "possible pharmacological intervention and crisis monitoring."
        )