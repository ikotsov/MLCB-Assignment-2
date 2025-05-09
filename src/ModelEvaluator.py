import numpy as np
from sklearn.metrics import f1_score


class ModelEvaluator:
    """
    Utility class for evaluating a pipeline using CV and a scoring function.
    """

    @staticmethod
    def evaluate(pipeline, X, y, cv, scorer=f1_score):
        """
        Evaluate a pipeline using cross-validation.

        Args:
            pipeline: A scikit-learn pipeline.
            X (array-like): Features.
            y (array-like): Labels.
            cv: A scikit-learn CV splitter (e.g., StratifiedKFold).
            scorer (callable): Scoring function (default: f1_score).

        Returns:
            float: Mean score across all CV folds.
        """
        scores = []
        for train_idx, val_idx in cv.split(X, y):
            pipeline.fit(X[train_idx], y[train_idx])
            y_pred = pipeline.predict(X[val_idx])
            scores.append(scorer(y[val_idx], y_pred))
        return np.mean(scores)
