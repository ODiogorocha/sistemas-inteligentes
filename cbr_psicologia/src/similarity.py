"""
Similaridade clínica para CBR (numérica + categórica + textual)
"""

FEATURE_WEIGHTS = {

    "gad7_estimate": 5.0,
    "phq9_estimate": 5.0,
    "clinical_severity": 4.0,
    "work_or_study_impairment": 4.0,

    "stress_level": 3.0,
    "panic_symptoms": 2.5,
    "sleep_quality": 2.0,
    "anxiety_score": 2.0,
    "depression_score": 2.0,

    "social_support": 1.5,
    "physical_activity": 1.0,
    "concentration_difficulty": 1.0,

    # 🔥 IMPORTANTE: descrição entra no CBR
    "main_issue": 2.0,

    "age": 0.0,
    "gender": 0.0,
    "bmi_estimate": 0.0,
    "sleep_hours": 0.0,
    "symptom_duration_months": 0.0,
    "irritability_level": 0.0,
    "appetite_change": 0.0,
    "prior_treatment": 0.0,
    "current_medication": 0.0,
    "trauma_history": 0.0,
    "substance_use_risk": 0.0,
    "comorbid_profile": 0.0,
}


SEVERITY_MATRIX = {
    ("mild", "mild"): 1.0,
    ("mild", "moderate"): 0.5,
    ("mild", "severe"): 0.0,

    ("moderate", "mild"): 0.5,
    ("moderate", "moderate"): 1.0,
    ("moderate", "severe"): 0.5,

    ("severe", "mild"): 0.0,
    ("severe", "moderate"): 0.5,
    ("severe", "severe"): 1.0,
}


IMPAIRMENT_MATRIX = {
    ("low", "low"): 1.0,
    ("low", "moderate"): 0.5,
    ("low", "high"): 0.0,

    ("moderate", "low"): 0.5,
    ("moderate", "moderate"): 1.0,
    ("moderate", "high"): 0.5,

    ("high", "low"): 0.0,
    ("high", "moderate"): 0.5,
    ("high", "high"): 1.0,
}


def text_similarity(a, b):

    if not a or not b:
        return 0.0

    a = str(a).lower().split()
    b = str(b).lower().split()

    if not a or not b:
        return 0.0

    return len(set(a) & set(b)) / len(set(a) | set(b))


def numeric_similarity(a, b, min_val, max_val):

    try:
        a = float(a)
        b = float(b)
    except:
        return 0.0

    if max_val == min_val:
        return 1.0

    return max(0.0, 1.0 - abs(a - b) / (max_val - min_val))


def categorical_similarity(a, b, matrix=None):

    a = str(a).lower().strip()
    b = str(b).lower().strip()

    if matrix:
        return matrix.get((a, b), 0.0)

    return 1.0 if a == b else 0.0


def _select(key, v1, v2, ranges):

    weight = FEATURE_WEIGHTS.get(key, 1.0)

    if weight <= 0:
        return 0.0, 0.0

    if isinstance(v1, (int, float)):
        r = ranges.get(key, {"min": 0, "max": 1})
        return numeric_similarity(v1, v2, r["min"], r["max"]), weight

    if key == "clinical_severity":
        return categorical_similarity(v1, v2, SEVERITY_MATRIX), weight

    if key == "work_or_study_impairment":
        return categorical_similarity(v1, v2, IMPAIRMENT_MATRIX), weight

    return categorical_similarity(v1, v2), weight


def compute_similarity(case_problem, query, ranges):

    total = 0
    weight_sum = 0

    for k, v1 in case_problem.items():

        if k == "case_id":
            continue

        if k not in query:
            continue

        v2 = query[k]

        if v1 is None or v2 is None:
            continue

        score, w = _select(k, v1, v2, ranges)

        total += score * w
        weight_sum += w

    # 🔥 bônus textual forte
    if "main_issue" in case_problem and "main_issue" in query:

        total += text_similarity(
            case_problem["main_issue"],
            query["main_issue"]
        ) * 2.0

        weight_sum += 2.0

    if weight_sum == 0:
        return 0.0

    return total / weight_sum