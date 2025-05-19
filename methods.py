import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import (
    mutual_info_classif, SelectKBest,
    chi2, f_classif,
    RFECV as SklearnRFECV,
    SelectFromModel,
    SequentialFeatureSelector as SklearnSequentialFS
)
from sklearn.linear_model import LogisticRegression, LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import SparsePCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from boruta import BorutaPy


def _validate_array(X):
    """
    Ensure X is a NumPy array.
    """
    return np.asarray(X)


class _BaseSelector(BaseEstimator, TransformerMixin):
    """
    Abstract base for wrapping sklearn feature selectors or custom logic.
    """
    def __init__(self):
        self.rng = None

    def fit(self, X, y=None):
        raise NotImplementedError

    def transform(self, X):
        raise NotImplementedError

    def fit_transform(self, X, y=None, **kwargs):
        return self.fit(X, y, **kwargs).transform(X)


class MISelector(_BaseSelector):
    """
    Select top-k features based on mutual information.
    """
    def __init__(self, k: int = 10):
        super().__init__()
        self.k = k
        self._selector = SelectKBest(mutual_info_classif, k=self.k)

    def fit(self, X, y):
        X_arr = _validate_array(X)
        self._selector.fit(X_arr, y)
        return self

    def transform(self, X):
        return self._selector.transform(_validate_array(X))


class Chi2Selector(_BaseSelector):
    """
    Select top-k features based on chi-squared test.
    """
    def __init__(self, k: int = 10):
        super().__init__()
        self.k = k
        self._selector = SelectKBest(chi2, k=self.k)

    def fit(self, X, y):
        self._selector.fit(_validate_array(X), y)
        return self

    def transform(self, X):
        return self._selector.transform(_validate_array(X))


class ANOVAFSelector(_BaseSelector):
    """
    Select top-k features based on ANOVA F-test.
    """
    def __init__(self, k: int = 10):
        super().__init__()
        self.k = k
        self._selector = SelectKBest(f_classif, k=self.k)

    def fit(self, X, y):
        self._selector.fit(_validate_array(X), y)
        return self

    def transform(self, X):
        return self._selector.transform(_validate_array(X))


class LassoCVSelector(_BaseSelector):
    """
    Use LassoCV to select features with non-zero coefficients.
    """
    def __init__(
        self,
        alphas: np.ndarray = np.logspace(-4, 1, 50),
        cv: int = 5,
        random_state: int = None
    ):
        super().__init__()
        self.alphas = alphas
        self.cv = cv
        self.random_state = random_state
        self.model: LassoCV = None

    def fit(self, X, y):
        self.rng = check_random_state(self.random_state)
        X_arr = _validate_array(X)
        self.model = LassoCV(
            alphas=self.alphas,
            cv=self.cv,
            random_state=self.random_state
        ).fit(X_arr, y)
        self._mask = self.model.coef_ != 0
        return self

    def transform(self, X):
        X_arr = _validate_array(X)
        return X_arr[:, self._mask]


class RandomForestSelector(_BaseSelector):
    """
    Rank features by RandomForest importance and pick top-n.
    """
    def __init__(
        self,
        n_estimators: int = 100,
        max_features: str = 'auto',
        n_selected: int = 10,
        random_state: int = None
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.n_selected = n_selected
        self.random_state = random_state

    def fit(self, X, y):
        self.rng = check_random_state(self.random_state)
        X_arr = _validate_array(X)
        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_features=self.max_features,
            random_state=self.random_state
        ).fit(X_arr, y)
        importances = model.feature_importances_
        self._mask = np.argsort(importances)[-self.n_selected:]
        return self

    def transform(self, X):
        return _validate_array(X)[:, self._mask]


class ElasticNetCVSelector(_BaseSelector):
    """
    Use ElasticNetCV to select features with non-zero coefficients.
    """
    def __init__(
        self,
        cv: int = 5,
        l1_ratio: float = 0.5,
        max_iter: int = 1000,
        random_state: int = None
    ):
        super().__init__()
        self.cv = cv
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        self.rng = check_random_state(self.random_state)
        X_arr = _validate_array(X)
        self.model = ElasticNetCV(
            cv=self.cv,
            l1_ratio=self.l1_ratio,
            max_iter=self.max_iter,
            random_state=self.random_state
        ).fit(X_arr, y)
        self._mask = self.model.coef_ != 0
        return self

    def transform(self, X):
        return _validate_array(X)[:, self._mask]


class SparsePCATransformer(_BaseSelector):
    """
    Reduce dimensionality via SparsePCA.
    """
    def __init__(
        self,
        n_components: int = 100,
        alpha: float = 1.0,
        random_state: int = None
    ):
        super().__init__()
        self.n_components = n_components
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, X, y=None):
        X_arr = _validate_array(X)
        self.model = SparsePCA(
            n_components=self.n_components,
            alpha=self.alpha,
            random_state=self.random_state
        ).fit(X_arr)
        return self

    def transform(self, X):
        return self.model.transform(_validate_array(X))


class StabilityFeatureSelector(_BaseSelector):
    """
    Stability-based selection by counting significant f-test events.
    """
    def __init__(
        self,
        iterations: int = 100,
        fraction: float = 0.75,
        threshold: float = 0.5,
        n_selected: int = 10,
        random_state: int = None
    ):
        super().__init__()
        self.iterations = iterations
        self.fraction = fraction
        self.threshold = threshold
        self.n_selected = n_selected
        self.random_state = random_state

    def fit(self, X, y):
        self.rng = check_random_state(self.random_state)
        X_arr, y_arr = _validate_array(X), np.asarray(y)
        n_samples, n_features = X_arr.shape
        scores = np.zeros(n_features)

        for _ in range(self.iterations):
            mask = self.rng.rand(n_samples) < self.fraction
            sel = SelectKBest(f_classif, k='all').fit(X_arr[mask], y_arr[mask])
            median_score = np.median(sel.scores_)
            scores += (sel.scores_ > median_score).astype(int)

        freq = scores / self.iterations
        mask_sel = freq > self.threshold
        if mask_sel.sum() < self.n_selected:
            mask_sel[np.argsort(freq)[-self.n_selected:]] = True

        self._mask = mask_sel
        return self

    def transform(self, X):
        return _validate_array(X)[:, self._mask]


class EnsembleVotingSelector(_BaseSelector):
    """
    Aggregate feature importances from multiple models via voting.
    """
    def __init__(
        self,
        estimators=None,
        n_selected: int = 10,
        voting: str = 'soft'
    ):
        super().__init__()
        self.voting = voting
        self.n_selected = n_selected
        self.estimators = estimators or [
            ('lasso', LassoCV(random_state=42)),
            ('rf', RandomForestClassifier(random_state=42)),
            ('lr', LogisticRegression(random_state=42))
        ]

    def fit(self, X, y):
        X_arr, y_arr = _validate_array(X), np.asarray(y)
        importances = np.zeros(X_arr.shape[1])

        for _, est in self.estimators:
            selector = SelectFromModel(est, max_features=self.n_selected).fit(X_arr, y_arr)
            if self.voting == 'soft':
                if hasattr(selector.estimator_, 'coef_'):
                    importances += np.abs(selector.estimator_.coef_).ravel()
                else:
                    importances += selector.estimator_.feature_importances_
            elif self.voting == 'hard':
                importances += selector.get_support().astype(int)

        top_idx = np.argsort(importances)[-self.n_selected:]
        mask = np.zeros_like(importances, dtype=bool)
        mask[top_idx] = True
        self._mask = mask
        return self

    def transform(self, X):
        return _validate_array(X)[:, self._mask]


class BorutaFeatureSelector(_BaseSelector):
    """
    Wrap BorutaPy for robust feature filtering.
    """
    def __init__(
        self,
        n_estimators='auto',
        perc: int = 100,
        alpha: float = 0.05,
        max_iter: int = 100,
        random_state: int = None
    ):
        super().__init__()
        self.params = dict(
            n_estimators=n_estimators,
            perc=perc,
            alpha=alpha,
            max_iter=max_iter,
            random_state=random_state
        )

    def fit(self, X, y):
        X_arr, y_arr = _validate_array(X), np.asarray(y)
        rf = RandomForestClassifier(
            n_estimators=self.params['n_estimators'],
            random_state=self.params['random_state']
        )
        self.boruta = BorutaPy(
            rf,
            n_estimators=self.params['n_estimators'],
            perc=self.params['perc'],
            alpha=self.params['alpha'],
            max_iter=self.params['max_iter'],
            random_state=self.params['random_state']
        )
        self.boruta.fit(X_arr, y_arr)
        self._mask = self.boruta.support_
        return self

    def transform(self, X):
        return _validate_array(X)[:, self._mask]


class ReliefFSelector(_BaseSelector):
    """
    ReliefF algorithm for feature weighting and selection.
    """
    def __init__(
        self,
        n_neighbors: int = 10,
        n_selected: int = 10,
        random_state: int = None
    ):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.n_selected = n_selected
        self.random_state = random_state

    def fit(self, X, y):
        self.rng = check_random_state(self.random_state)
        X_arr, y_arr = StandardScaler().fit_transform(
            _validate_array(X)
        ), np.asarray(y)
        n_samples, n_features = X_arr.shape
        weights = np.zeros(n_features)

        for _ in range(n_samples):
            i = self.rng.randint(n_samples)
            same = np.where(y_arr == y_arr[i])[0]
            diff = np.where(y_arr != y_arr[i])[0]
            # find nearest hit/miss
            hit = same[np.argsort(np.sum((X_arr[same] - X_arr[i])**2, axis=1))[1]]
            miss = diff[np.argmin(np.sum((X_arr[diff] - X_arr[i])**2, axis=1))]
            weights += np.abs(X_arr[i] - X_arr[hit]) - np.abs(X_arr[i] - X_arr[miss])

        self.feature_scores_ = weights / n_samples
        top_idx = np.argsort(self.feature_scores_)[-self.n_selected:]
        mask = np.zeros(n_features, dtype=bool)
        mask[top_idx] = True
        self._mask = mask
        return self

    def transform(self, X):
        return _validate_array(X)[:, self._mask]


class RecursiveEliminationCVSelector(_BaseSelector):
    """
    RFECV-based recursive feature elimination.
    """
    def __init__(
        self,
        estimator=None,
        step: int = 1,
        cv: int = 5,
        scoring: str = 'accuracy',
        n_jobs: int = -1
    ):
        super().__init__()
        self.estimator = estimator or LogisticRegression(random_state=42)
        self.step = step
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.selector = None

    def fit(self, X, y):
        self.selector = SklearnRFECV(
            estimator=self.estimator,
            step=self.step,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs
        ).fit(_validate_array(X), y)
        self._mask = self.selector.support_
        return self

    def transform(self, X):
        return _validate_array(X)[:, self._mask]


class SequentialSFSSelector(_BaseSelector):
    """
    Forward/backward sequential feature selection.
    """
    def __init__(
        self,
        estimator=None,
        n_selected: int = 10,
        direction: str = 'forward',
        scoring: str = 'accuracy',
        cv: int = 5,
        n_jobs: int = -1
    ):
        super().__init__()
        self.estimator = estimator or LogisticRegression(random_state=42)
        self.n_selected = n_selected
        self.direction = direction
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.selector = None

    def fit(self, X, y):
        self.selector = SklearnSequentialFS(
            estimator=self.estimator,
            n_features_to_select=self.n_selected,
            direction=self.direction,
            scoring=self.scoring,
            cv=self.cv,
            n_jobs=self.n_jobs
        ).fit(_validate_array(X), y)
        self._mask = self.selector.get_support()
        return self

    def transform(self, X):
        return _validate_array(X)[:, self._mask]
