"""
Módulo de similaridade para o sistema CBR de psicologia.
Implementa medidas locais distintas para atributos numéricos,
categóricos e textuais, conforme requisito do trabalho.
"""

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import cbrkit.sim.numbers as num_sim

# ------------------------------------------------------------------
# Configuração de pesos por feature
# Atributos mais relevantes clinicamente recebem peso maior.
# ------------------------------------------------------------------
FEATURE_WEIGHTS = {
    # Numéricos
    "age":                       0.5,
    "anxiety_score":             1.5,
    "depression_score":          1.5,
    "stress_level":              1.2,
    "sleep_quality":             1.0,
    "sleep_hours":               0.8,
    "symptom_duration_months":   1.0,
    "gad7_estimate":             1.5,
    "phq9_estimate":             1.5,
    "irritability_level":        1.0,
    "bmi_estimate":              0.5,
    # Categóricos
    "gender":                    0.5,
    "social_support":            1.0,
    "physical_activity":         0.8,
    "panic_symptoms":            1.2,
    "concentration_difficulty":  1.0,
    "appetite_change":           0.8,
    "prior_treatment":           1.0,
    "current_medication":        1.0,
    "trauma_history":            1.2,
    "substance_use_risk":        1.0,
    "work_or_study_impairment":  1.0,
    "comorbid_profile":          1.0,
    "clinical_severity":         1.5,
    # Textual
    "main_issue":                1.2,
}

# Matrizes de similaridade para atributos ordinais / com semântica
SEVERITY_MATRIX = {
    ("mild",     "mild"):     1.0,
    ("mild",     "moderate"): 0.5,
    ("mild",     "severe"):   0.0,
    ("moderate", "mild"):     0.5,
    ("moderate", "moderate"): 1.0,
    ("moderate", "severe"):   0.5,
    ("severe",   "mild"):     0.0,
    ("severe",   "moderate"): 0.5,
    ("severe",   "severe"):   1.0,
}

SUPPORT_MATRIX = {
    ("low",    "low"):    1.0,
    ("low",    "medium"): 0.5,
    ("low",    "high"):   0.0,
    ("medium", "low"):    0.5,
    ("medium", "medium"): 1.0,
    ("medium", "high"):   0.5,
    ("high",   "low"):    0.0,
    ("high",   "medium"): 0.5,
    ("high",   "high"):   1.0,
}

ACTIVITY_MATRIX = {
    ("low",      "low"):      1.0,
    ("low",      "moderate"): 0.5,
    ("low",      "high"):     0.0,
    ("moderate", "low"):      0.5,
    ("moderate", "moderate"): 1.0,
    ("moderate", "high"):     0.5,
    ("high",     "low"):      0.0,
    ("high",     "moderate"): 0.5,
    ("high",     "high"):     1.0,
}

IMPAIRMENT_MATRIX = {
    ("low",      "low"):      1.0,
    ("low",      "moderate"): 0.5,
    ("low",      "high"):     0.0,
    ("moderate", "low"):      0.5,
    ("moderate", "moderate"): 1.0,
    ("moderate", "high"):     0.5,
    ("high",     "low"):      0.0,
    ("high",     "moderate"): 0.5,
    ("high",     "high"):     1.0,
}

# Simulador linear do cbrkit (normalizado 0-1)
_linear = num_sim.linear_interval(min=0.0, max=1.0)


def numeric_similarity(a, b, min_val, max_val):
    """Similaridade normalizada por distância linear usando cbrkit."""
    if max_val == min_val:
        return 1.0
    a_norm = (float(a) - min_val) / (max_val - min_val)
    b_norm = (float(b) - min_val) / (max_val - min_val)
    return float(_linear(a_norm, b_norm))


def categorical_similarity(a, b, matrix=None):
    """
    Similaridade categórica.
    Se matriz fornecida, usa ela (atributos ordinais/semânticos).
    Caso contrário, aplica igualdade estrita.
    """
    a_s = str(a).strip().lower()
    b_s = str(b).strip().lower()
    if matrix is not None:
        return matrix.get((a_s, b_s), matrix.get((b_s, a_s), 0.0))
    return 1.0 if a_s == b_s else 0.0


def text_similarity(a, b):
    """Similaridade coseno com TF-IDF para atributos textuais."""
    a_s = str(a).strip()
    b_s = str(b).strip()
    if not a_s or not b_s:
        return 0.0
    if a_s.lower() == b_s.lower():
        return 1.0
    try:
        vect = TfidfVectorizer()
        tfidf = vect.fit_transform([a_s, b_s])
        score = cosine_similarity(tfidf[0], tfidf[1])[0][0]
        return float(score)
    except Exception:
        return 0.0


def _select_sim(key, v1, v2, ranges):
    """
    Seleciona e executa a função de similaridade correta
    com base no nome e tipo da feature.
    Retorna (sim_score, weight).
    """
    weight = FEATURE_WEIGHTS.get(key, 1.0)

    # --- Numérico ---
    if isinstance(v1, (int, float)):
        try:
            v2f = float(v2)
        except (ValueError, TypeError):
            return 0.0, weight
        r = ranges.get(key, {})
        score = numeric_similarity(v1, v2f, r.get("min", 0), r.get("max", 1))
        return score, weight

    # --- Textual ---
    if key == "main_issue":
        return text_similarity(v1, v2), weight

    # --- Categóricos com matriz semântica ---
    if key == "clinical_severity":
        return categorical_similarity(v1, v2, SEVERITY_MATRIX), weight
    if key == "social_support":
        return categorical_similarity(v1, v2, SUPPORT_MATRIX), weight
    if key == "physical_activity":
        return categorical_similarity(v1, v2, ACTIVITY_MATRIX), weight
    if key == "work_or_study_impairment":
        return categorical_similarity(v1, v2, IMPAIRMENT_MATRIX), weight

    # --- Categórico genérico (igualdade estrita) ---
    return categorical_similarity(v1, v2), weight


def compute_similarity(case_problem, query, ranges):
    """
    Similaridade global ponderada entre um caso da base e a query.
    Usa média ponderada das similaridades locais por feature.
    """
    total_weight = 0.0
    weighted_sum = 0.0

    for key, v1 in case_problem.items():
        if key == "case_id":
            continue
        v2 = query.get(key, v1)
        score, weight = _select_sim(key, v1, v2, ranges)
        weighted_sum += score * weight
        total_weight += weight

    if total_weight == 0:
        return 0.0
    return weighted_sum / total_weight