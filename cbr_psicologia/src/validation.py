"""
Métodos de validação do sistema CBR:
  - Leave-One-Out (LOO)
  - K-Fold Cross-Validation
"""
import random
from retrieval import retrieve_cases
from adaptation import adapt_solution


def leave_one_out(casebase, ranges, k=3, verbose=False):
    """
    Validação Leave-One-Out:
    Cada caso é removido da base, usado como query e avaliado.

    Args:
        casebase: lista completa de casos
        ranges:   dict com min/max das features numéricas
        k:        número de vizinhos usados na recuperação
        verbose:  se True, imprime progresso caso a caso

    Returns:
        y_true, y_pred       → intervention_type (str)
        true_int, pred_int   → intensity (float)
        true_freq, pred_freq → weekly_frequency (float)
    """
    y_true, y_pred   = [], []
    true_int, pred_int   = [], []
    true_freq, pred_freq = [], []

    n = len(casebase)
    for i in range(n):
        test  = casebase[i]
        train = casebase[:i] + casebase[i + 1:]

        retrieved = retrieve_cases(train, test["problem"], k, ranges)
        pred      = adapt_solution(retrieved, test["problem"])

        y_true.append(test["solution"]["intervention_type"])
        y_pred.append(pred["intervention_type"])

        true_int.append(float(test["solution"]["intensity"]))
        pred_int.append(float(pred["intensity"]))

        true_freq.append(float(test["solution"]["weekly_frequency"]))
        pred_freq.append(float(pred["weekly_frequency"]))

        if verbose:
            print(f"  [{i+1}/{n}] true={y_true[-1]!r:15} pred={y_pred[-1]!r:15} "
                  f"int={true_int[-1]:.0f}/{pred_int[-1]:.1f}")

    return y_true, y_pred, true_int, pred_int, true_freq, pred_freq


def kfold_cross_validation(casebase, ranges, k_folds=5, k_neighbors=3):
    """
    Validação K-Fold:
    Divide o casebase em k_folds partes; cada fold é testado uma vez.

    Args:
        casebase:    lista completa de casos
        ranges:      dict com min/max das features numéricas
        k_folds:     número de folds
        k_neighbors: número de vizinhos usados na recuperação

    Returns:
        y_true, y_pred       → intervention_type (str)
        true_int, pred_int   → intensity (float)
        true_freq, pred_freq → weekly_frequency (float)
    """
    shuffled = casebase.copy()
    random.shuffle(shuffled)

    folds = [shuffled[i::k_folds] for i in range(k_folds)]

    y_true, y_pred   = [], []
    true_int, pred_int   = [], []
    true_freq, pred_freq = [], []

    for fold_idx in range(k_folds):
        test_fold  = folds[fold_idx]
        train_fold = []
        for j in range(k_folds):
            if j != fold_idx:
                train_fold.extend(folds[j])

        for test in test_fold:
            retrieved = retrieve_cases(train_fold, test["problem"], k_neighbors, ranges)
            pred      = adapt_solution(retrieved, test["problem"])

            y_true.append(test["solution"]["intervention_type"])
            y_pred.append(pred["intervention_type"])

            true_int.append(float(test["solution"]["intensity"]))
            pred_int.append(float(pred["intensity"]))

            true_freq.append(float(test["solution"]["weekly_frequency"]))
            pred_freq.append(float(pred["weekly_frequency"]))

    return y_true, y_pred, true_int, pred_int, true_freq, pred_freq