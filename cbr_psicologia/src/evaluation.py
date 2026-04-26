"""
Módulo de avaliação: calcula acurácia, precisão, recall, F1
e erro médio absoluto (MAE) para intensity e weekly_frequency.
"""
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate(y_true, y_pred):
    """
    Métricas de classificação para intervention_type.

    Returns:
        Dict com accuracy, precision, recall e f1 (macro).
    """
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall":    recall_score(y_true, y_pred,    average="macro", zero_division=0),
        "f1":        f1_score(y_true, y_pred,         average="macro", zero_division=0),
    }


def adaptation_error(true_vals, pred_vals):
    """
    Erro médio absoluto (MAE) entre intensity verdadeira e predita.
    """
    if not true_vals:
        return 0.0
    return sum(abs(float(t) - float(p)) for t, p in zip(true_vals, pred_vals)) / len(true_vals)


def freq_error(true_vals, pred_vals):
    """
    Erro médio absoluto (MAE) de weekly_frequency.
    """
    if not true_vals:
        return 0.0
    return sum(abs(float(t) - float(p)) for t, p in zip(true_vals, pred_vals)) / len(true_vals)