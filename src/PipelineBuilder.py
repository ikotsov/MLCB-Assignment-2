from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class PipelineBuilder:
    """
    Utility class to build a consistent preprocessing pipeline for classification tasks.
    """

    @staticmethod
    def create(estimator):
        """
        Builds a pipeline with imputation, scaling, PCA, and the given estimator.

        Args:
            estimator: A scikit-learn estimator instance (e.g., LogisticRegression()).

        Returns:
            Pipeline: A scikit-learn pipeline.
        """
        return Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.95)),  # Adjust as needed
            ('clf', estimator)
        ])
