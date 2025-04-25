import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import matthews_corrcoef, roc_auc_score, balanced_accuracy_score, f1_score, fbeta_score, recall_score, precision_score
import optuna
from optuna.samplers import TPESampler

# Number of trials for Optuna
NUM_TRIALS = 50
# Number of rounds for the nCV
DEFAULT_R = 10
# Number of outer fold loops
DEFAULT_N = 5
# Number of inner fold loops
DEFAULT_K = 3
DEFAULT_SEED = 42


class RepeatedNestedCV:
    def __init__(self, X, y, estimators, param_spaces, R=DEFAULT_R, N=DEFAULT_N, K=DEFAULT_K, seed=DEFAULT_SEED):
        self.X = X
        self.y = y
        self.estimators = estimators
        self.param_spaces = param_spaces
        self.R = R
        self.N = N
        self.K = K
        self.seed = seed
        self.results = []

    def run(self):
        self.clear_results()  # Reset results on each call

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
                    pipeline = self._build_pipeline(
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
        # Suggest a new set of hyperparameters for this trial
        params = param_space(trial)

        pipeline = self._build_pipeline(estimator_class(**params))

        inner_cv = StratifiedKFold(
            n_splits=self.K, shuffle=True, random_state=self.seed + rep)

        scores = []
        for train_subset, val_subset in inner_cv.split(X_train, y_train):
            # Train on inner train split, evaluate on inner validation split
            pipeline.fit(X_train[train_subset], y_train[train_subset])
            y_pred = pipeline.predict(X_train[val_subset])
            score = f1_score(y_train[val_subset], y_pred)
            scores.append(score)

        return np.mean(scores)

    def _build_pipeline(self, estimator):
        return Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=7)),
            ('clf', estimator)
        ])

    def _compute_metrics(self, y_true, y_pred, y_prob=None):
        return {
            'MCC': matthews_corrcoef(y_true, y_pred),
            'AUC': roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan,
            'BA': balanced_accuracy_score(y_true, y_pred),
            'F1': f1_score(y_true, y_pred),
            'F2': fbeta_score(y_true, y_pred, beta=2),
            'Recall': recall_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred)
        }
