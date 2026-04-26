"""
Módulo de adaptação das soluções recuperadas.
Combina múltiplos casos recuperados e aplica regras clínicas
para ajustar a solução — não copia apenas o caso mais similar.
"""

# ---------------------------------------------------------
# Ajuste de intensidade por severidade clínica
# ---------------------------------------------------------
SEVERITY_INTENSITY = {
    "mild":     -1,
    "moderate":  0,
    "severe":   +1,
}

# Complementos de texto por perfil clínico
EXTRA_RECOMMENDATIONS = {
    "trauma":      "Considere abordagem sensível ao trauma (TCC-focalizada no trauma ou EMDR).",
    "substance":   "Inclua avaliação de risco de uso de substâncias e orientação motivacional.",
    "panic":       "Inclua técnicas de regulação autonômica e psicoeducação sobre pânico.",
    "sleep":       "Protocolo de higiene do sono e TCC para insônia (CBT-I) recomendado.",
    "social":      "Estimular participação social gradual e atividades em grupo.",
}


def _weighted_vote(retrieved_cases):
    """
    Combina as soluções dos k casos recuperados usando:
    - Votação ponderada pela similaridade para intervention_type
    - Média ponderada para intensity e weekly_frequency
    """
    votes = {}
    total_sim = 0.0
    weighted_intensity = 0.0
    weighted_freq = 0.0
    texts = []

    for sim, case in retrieved_cases:
        sol = case["solution"]
        itype = sol.get("intervention_type", "")
        votes[itype] = votes.get(itype, 0.0) + sim
        total_sim += sim
        weighted_intensity += sim * float(sol.get("intensity", 3))
        weighted_freq += sim * float(sol.get("weekly_frequency", 2))
        texts.append(sol.get("recommendation_text", ""))

    if total_sim == 0:
        total_sim = 1.0

    best_type    = max(votes, key=votes.get)
    avg_intensity = weighted_intensity / total_sim
    avg_freq      = weighted_freq / total_sim

    return best_type, avg_intensity, avg_freq, texts


def adapt_solution(retrieved_cases, new_case):
    """
    Adapta a solução ao novo caso aplicando:
    1. Votação ponderada entre os k casos recuperados
    2. Ajuste de intensidade pela severidade clínica
    3. Ajuste de frequência por comprometimento funcional
    4. Ajuste adicional por pontuações GAD-7 / PHQ-9 altas
    5. Complementos textuais automáticos por perfil clínico

    Args:
        retrieved_cases: lista de (sim_score, case) do retrieval
        new_case:        dict com features do novo problema

    Returns:
        Dict com intervention_type, intensity, weekly_frequency,
        recommendation_text adaptados.
    """
    if not retrieved_cases:
        return {
            "intervention_type": "psychotherapy",
            "intensity": 3,
            "weekly_frequency": 2,
            "recommendation_text": "Avaliação clínica detalhada recomendada.",
        }

    intervention_type, intensity, weekly_frequency, texts = _weighted_vote(retrieved_cases)

    # --- Ajuste por severidade clínica ---
    severity = str(new_case.get("clinical_severity", "moderate")).strip().lower()
    intensity += SEVERITY_INTENSITY.get(severity, 0)

    # Se severo e intervenção fraca → promove para psychotherapy
    if severity == "severe" and intervention_type in ("exercise", "psychoeducation"):
        intervention_type = "psychotherapy"

    # --- Ajuste por comprometimento funcional (categórico: low/moderate/high) ---
    impairment = str(new_case.get("work_or_study_impairment", "moderate")).strip().lower()
    if impairment == "high":
        weekly_frequency = min(weekly_frequency + 1, 7)

    # --- Ajuste por pontuações clínicas altas ---
    gad7 = float(new_case.get("gad7_estimate", 0))
    phq9 = float(new_case.get("phq9_estimate", 0))
    if gad7 >= 15 or phq9 >= 15:
        intensity = min(intensity + 0.5, 5)

    # Clampa intensity e weekly_frequency nos limites válidos
    intensity        = round(max(1.0, min(5.0, intensity)), 1)
    weekly_frequency = round(max(1.0, min(7.0, weekly_frequency)), 1)

    # --- Texto base: do caso mais similar ---
    base_text = texts[0] if texts else ""

    # --- Complementos textuais por perfil ---
    extras = []
    if str(new_case.get("trauma_history", "")).strip().lower() not in ("none", "no", ""):
        extras.append(EXTRA_RECOMMENDATIONS["trauma"])
    if str(new_case.get("substance_use_risk", "")).strip().lower() not in ("none", "no", ""):
        extras.append(EXTRA_RECOMMENDATIONS["substance"])
    if str(new_case.get("panic_symptoms", "")).strip().lower() == "yes":
        extras.append(EXTRA_RECOMMENDATIONS["panic"])
    if float(new_case.get("sleep_quality", 5)) <= 3:
        extras.append(EXTRA_RECOMMENDATIONS["sleep"])
    if str(new_case.get("social_support", "")).strip().lower() == "low":
        extras.append(EXTRA_RECOMMENDATIONS["social"])

    full_text = base_text
    if extras:
        full_text += " | Adaptações: " + " ".join(extras)

    return {
        "intervention_type": intervention_type,
        "intensity":         intensity,
        "weekly_frequency":  weekly_frequency,
        "recommendation_text": full_text,
    }