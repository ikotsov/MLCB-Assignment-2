import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef, roc_auc_score, balanced_accuracy_score, f1_score, fbeta_score, recall_score, precision_score
import optuna
from optuna.samplers import TPESampler
from PipelineBuilder import PipelineBuilder
from ModelEvaluator import ModelEvaluator

# Suppress warnings to keep notebooks clean. Comment this out while debugging.
import warnings
warnings.filterwarnings('ignore')

# Suppress Optuna logging to avoid cluttering the output.
# optuna.logging.set_verbosity(optuna.logging.CRITICAL)

NUM_TRIALS = 50  # Number of trials for Optuna
DEFAULT_R = 10  # Number of rounds for the nCV
DEFAULT_N = 5  # Number of outer fold loops
DEFAULT_K = 3  # Number of inner fold loops
DEFAULT_SEED = 42


class RepeatedNestedCV:
    """
    This class implements a repeated nested cross-validation (rnCV) pipeline for model selection and performance estimation.
    It supports multiple estimators, hyperparameter optimization via Optuna and metric evaluation.
    """

    METRICS = ['MCC', 'AUC', 'BA', 'F1', 'F2', 'Recall', 'Precision']

    def __init__(self, X, y, estimators, param_spaces, R=DEFAULT_R, N=DEFAULT_N, K=DEFAULT_K, seed=DEFAULT_SEED):
        """
        Initialize the RepeatedNestedCV class.

        Args:
            X (array-like): Feature matrix of shape (n_samples, n_features).
            y (array-like): Target vector of shape (n_samples,).
            estimators (dict): Dictionary mapping estimator names to constructor callables.
            param_spaces (dict): Dictionary mapping estimator names to Optuna-compatible 
                hyperparameter search space functions.
            R (int, optional): Number of repetitions for repeated nested CV.
            N (int, optional): Number of outer folds for generalization evaluation.
            K (int, optional): Number of inner folds for hyperparameter tuning.
            seed (int, optional): Base random seed used to ensure reproducibility across folds 
                and tuning trials.
        """
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.estimators = estimators
        self.param_spaces = param_spaces
        self.R = R
        self.N = N
        self.K = K
        self.seed = seed
        self.results = []

    def run(self):
        self.clear_results()  # Reset results on each call.

        for r in range(self.R):
            print(f"\n>>> Repetition {r + 1}/{self.R}")

            outer_cv = StratifiedKFold(
                n_splits=self.N, shuffle=True, random_state=self.seed + r)

            for fold_subset, (train_subset, test_subset) in enumerate(outer_cv.split(self.X, self.y)):
                print(f"\nOuter Fold {fold_subset + 1}/{self.N}")
                X_train, y_train = self.X[train_subset], self.y[train_subset]
                X_test, y_test = self.X[test_subset], self.y[test_subset]

                for name in self.estimators:
                    print(f"Tuning: {name}")
                    study = optuna.create_study(
                        direction="maximize", sampler=TPESampler(seed=self.seed + r))
                    study.optimize(lambda trial: self._objective(
                        trial, name, X_train, y_train, r), n_trials=NUM_TRIALS)

                    best_params = study.best_params
                    estimator_cls = self.estimators[name]
                    pipeline = PipelineBuilder.create(
                        estimator_cls(**best_params))
                    pipeline.fit(X_train, y_train)
                    y_pred = pipeline.predict(X_test)
                    y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(
                        pipeline, "predict_proba") else None

                    metrics = self._compute_metrics(y_test, y_pred, y_prob)

                    self.results.append({
                        'repetition': r + 1,
                        'outer_fold': fold_subset + 1,
                        'estimator': name,
                        **metrics
                    })

        return self.results

    def get_results(self):
        return self.results

    def clear_results(self):
        self.results = []

    def _objective(self, trial, estimator_name, X_train, y_train, rep):
        estimator_class = self.estimators[estimator_name]
        param_space = self.param_spaces[estimator_name]
        # Suggest a new set of hyperparameters for this trial.
        params = param_space(trial)

        pipeline = PipelineBuilder.create(estimator_class(**params))

        inner_cv = StratifiedKFold(
            n_splits=self.K, shuffle=True, random_state=self.seed + rep)

        return ModelEvaluator.evaluate(pipeline, X_train, y_train, inner_cv)

    def _compute_metrics(self, y_true, y_pred, y_prob=None):
        results = {}
        for name in self.METRICS:
            if name == 'MCC':
                results[name] = matthews_corrcoef(y_true, y_pred)
            elif name == 'AUC':
                # AUC requires probability estimates instead of discrete predictions.
                results[name] = roc_auc_score(
                    y_true, y_prob) if y_prob is not None else np.nan
            elif name == 'BA':
                results[name] = balanced_accuracy_score(y_true, y_pred)
            elif name == 'F1':
                results[name] = f1_score(y_true, y_pred)
            elif name == 'F2':
                # beta > 1, we care more about recall (we want fewer false negatives).
                results[name] = fbeta_score(y_true, y_pred, beta=2)
            elif name == 'Recall':
                results[name] = recall_score(y_true, y_pred)
            elif name == 'Precision':
                results[name] = precision_score(y_true, y_pred)
        return results
