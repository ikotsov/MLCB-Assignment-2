import numpy as np
import joblib
import os
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from PipelineBuilder import PipelineBuilder

DEFAULT_NUM_TRIALS = 50  # Number of trials for Optuna
DEFAULT_NUM_SPLITS = 5  # Number of folds for cross-validation
DEFAULT_SEED = 42


class ModelTuner:
    """
    This class performs hyperparameter tuning using CV, trains the model and saves it for deployment.
    """

    def __init__(self, X, y, estimator, param_space, n_splits=DEFAULT_NUM_SPLITS, seed=DEFAULT_SEED):
        """
        Args:
            X (array-like): Feature matrix.
            y (array-like): Target vector.
            estimator (callable): Estimator constructor (e.g., LogisticRegression, lambda **kwargs: SVC(...)).
            param_space (callable): Function that defines hyperparameter search space for Optuna.
            n_splits (int): Number of folds for cross-validation.
            seed (int): Random seed for reproducibility.
        """
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.estimator = estimator
        self.param_space = param_space
        self.n_splits = n_splits
        self.seed = seed

    def tune(self, n_trials=DEFAULT_NUM_TRIALS):
        """
        Tune hyperparameters using Optuna.

        Args:
            n_trials (int): Number of Optuna trials.

        Returns:
            dict: Best hyperparameters found.
        """
        study = optuna.create_study(direction="maximize")
        study.optimize(self._objective, n_trials=n_trials)
        return study.best_params

    def train(self, best_params):
        """
        Train the final model on the full dataset using best hyperparameters.

        Args:
            best_params (dict): Best hyperparameters to instantiate the estimator.

        Returns:
            sklearn.pipeline.Pipeline: Trained pipeline.
        """
        pipeline = PipelineBuilder.create(self.estimator(**best_params))
        pipeline.fit(self.X, self.y)
        return pipeline

    def save(self, model, path='./models/final_model.pkl'):
        """
        Save the trained model to a file.

        Args:
            model (Pipeline): Trained sklearn pipeline.
            path (str): File path to save the model.
        """
        os.makedirs(os.path.dirname(path),
                    exist_ok=True)  # Create folder if missing
        joblib.dump(model, path)

    def _objective(self, trial):
        """
        Objective function for Optuna hyperparameter tuning.
        """
        params = self.param_space(trial)
        pipeline = PipelineBuilder.create(self.estimator(**params))

        cv = StratifiedKFold(n_splits=self.n_splits,
                             shuffle=True, random_state=self.seed)

        scores = []
        for train_idx, val_idx in cv.split(self.X, self.y):
            pipeline.fit(self.X[train_idx], self.y[train_idx])
            y_pred = pipeline.predict(self.X[val_idx])
            score = f1_score(self.y[val_idx], y_pred)
            scores.append(score)

        return np.mean(scores)
