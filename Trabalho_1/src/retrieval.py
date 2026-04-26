def similarity(case1, case2):
    score = 0
    total_weight = 0

    
    weights = {
        "anxiety_score": 3,
        "depression_score": 3,
        "stress_level": 3,
        "sleep_quality": 2,
        "main_issue": 4,
        "clinical_severity": 3,
        "age": 1,
        "physical_activity": 1
    }

    for attr, weight in weights.items():
        val1 = getattr(case1, attr, None)
        val2 = getattr(case2, attr, None)

        if val1 is None or val2 is None:
            continue

        total_weight += weight

        # Numérico
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            diff = abs(val1 - val2)
            score += weight * (1 / (1 + diff))

        # Categórico
        else:
            if str(val1).lower() == str(val2).lower():
                score += weight

    return score / total_weight if total_weight > 0 else 0


def retrieve(new_case, case_base, k=3):
    scored = [(case, similarity(new_case, case)) for case in case_base]
    scored.sort(key=lambda x: x[1], reverse=True)

    return [case for case, _ in scored[:k]]