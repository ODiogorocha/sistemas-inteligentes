import random
from retrieval import retrieve_cases
from adaptation import adapt_solution


def leave_one_out(casebase, ranges, k=5, verbose=False):

    y_true = []
    y_pred = []

    n = len(casebase)

    for i in range(n):

        test = casebase[i]
        train = casebase[:i] + casebase[i+1:]

        retrieved = retrieve_cases(
            train,
            test["problem"],
            k,
            ranges
        )

        pred = adapt_solution(retrieved, test["problem"])

        y_true.append(test["problem"]["clinical_severity"])
        y_pred.append(pred["clinical_severity"])

        if verbose:
            print(f"[{i+1}/{n}] true={y_true[-1]} pred={y_pred[-1]}")

    return y_true, y_pred, [], [], [], []


def kfold_cross_validation(casebase, ranges, k_folds=5, k_neighbors=5):

    shuffled = casebase.copy()
    random.shuffle(shuffled)

    folds = [shuffled[i::k_folds] for i in range(k_folds)]

    y_true = []
    y_pred = []

    for i in range(k_folds):

        test_fold = folds[i]
        train_fold = []

        for j in range(k_folds):
            if j != i:
                train_fold.extend(folds[j])

        for test in test_fold:

            retrieved = retrieve_cases(
                train_fold,
                test["problem"],
                k_neighbors,
                ranges
            )

            pred = adapt_solution(retrieved, test["problem"])

            y_true.append(test["problem"]["clinical_severity"])
            y_pred.append(pred["clinical_severity"])

    return y_true, y_pred, [], [], [], []