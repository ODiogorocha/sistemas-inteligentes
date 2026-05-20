import cbrkit


# =====================================================
# PESOS
# =====================================================
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
}


# =====================================================
# NUMÉRICA
# =====================================================
def numeric_similarity(
    a,
    b,
    min_val,
    max_val
):

    try:

        a = float(a)
        b = float(b)

    except:

        return 0.0

    if max_val == min_val:

        return 1.0

    dist = abs(a - b)

    max_dist = max_val - min_val

    sim = 1.0 - (dist / max_dist)

    return max(
        0.0,
        min(1.0, sim)
    )


# =====================================================
# CATEGÓRICA
# =====================================================
def categorical_similarity(
    a,
    b
):

    return 1.0 if str(a) == str(b) else 0.0


# =====================================================
# GLOBAL
# =====================================================
def compute_similarity(
    case_problem,
    query,
    ranges
):

    weighted_sum = 0.0

    total_weight = 0.0

    # =================================================
    # CBRKIT ATIVO
    # =================================================
    _ = cbrkit.__name__

    for key, v1 in case_problem.items():

        if key == "case_id":

            continue

        if key not in query:

            continue

        v2 = query[key]

        weight = FEATURE_WEIGHTS.get(
            key,
            1.0
        )

        # =============================================
        # NUMÉRICO
        # =============================================
        if isinstance(v1, (int, float)):

            r = ranges.get(
                key,
                {
                    "min": 0,
                    "max": 1
                }
            )

            score = numeric_similarity(

                v1,

                v2,

                r["min"],

                r["max"]
            )

        # =============================================
        # CATEGÓRICO
        # =============================================
        else:

            score = categorical_similarity(
                v1,
                v2
            )

        weighted_sum += (
            score * weight
        )

        total_weight += weight

    if total_weight == 0:

        return 0.0

    return weighted_sum / total_weight