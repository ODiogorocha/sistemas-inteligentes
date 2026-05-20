from similarity import compute_similarity


def retrieve_cases(casebase, query, k, ranges):

    results = []

    for case in casebase:

        sim = compute_similarity(
            case["problem"],
            query,
            ranges
        )

        results.append((sim, case))

    results.sort(key=lambda x: x[0], reverse=True)

    # threshold adaptativo leve (não altera lógica base)
    if results:
        top_score = results[0][0]
        threshold = max(0.5, top_score - 0.15)

        filtered = [r for r in results if r[0] >= threshold]

        if filtered:
            return filtered[:k]

    return results[:k]