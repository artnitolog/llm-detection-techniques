import numpy as np
import sklearn.metrics as metrics


def eval_metrics(logits: np.ndarray, predicted_labels: np.ndarray, labels: np.ndarray):
    clf_metrics = dict(
        f1=metrics.f1_score(labels, predicted_labels),
        roc_auc=metrics.roc_auc_score(labels, logits),
        acc=metrics.accuracy_score(labels, predicted_labels),
        precision=metrics.precision_score(labels, predicted_labels),
        recall=metrics.recall_score(labels, predicted_labels),
        balanced_accuracy=metrics.balanced_accuracy_score(labels, predicted_labels),
    )
    return clf_metrics
