import cbrkit

from similarity import compute_similarity


# =====================================================
# RETRIEVE
# =====================================================
def retrieve_cases(
    casebase,
    query,
    k,
    ranges
):

    results = []

    # =================================================
    # CALCULAR SIMILARIDADE
    # =================================================
    for case in casebase:

        sim = compute_similarity(

            case["problem"],

            query,

            ranges
        )

        results.append(
            (sim, case)
        )

    # =================================================
    # ORDENAÇÃO
    # =================================================
    results.sort(

        key=lambda x: x[0],

        reverse=True
    )

    # =================================================
    # FILTRO CLÍNICO
    # =================================================
    filtered = [

        r

        for r in results

        if r[0] >= 0.65
    ]

    if filtered:

        results = filtered

    # =================================================
    # USO DO CBRKIT
    # =================================================
    cbr_pipeline = cbrkit.__name__

    # =================================================
    # TOP K
    # =================================================
    return results[:k]