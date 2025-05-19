import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif
from sklearn.utils import check_random_state


class Selector(BaseEstimator, TransformerMixin):
    """
    Monte Carlo Conformal Information-theoretic Feature Selector

    Parameters
    ----------
    alpha : float, default=0.2
        Significance level for conformal quantile.
    num_samples : int, default=100
        Number of Monte Carlo repetitions for importance estimation.
    random_state : int or None, default=None
        Seed for reproducible random number generation.
    """

    def __init__(self, alpha=0.2, num_samples=100, random_state=None):
        self.alpha = alpha
        self.num_samples = num_samples
        self.random_state = random_state
        self._rng = check_random_state(self.random_state)

    def _sample_mc_labels(self, class_probs, n_examples):
        """
        Draw Monte Carlo label samples for each repetition and example.

        Returns
        -------
        mc_labels : ndarray, shape (num_samples, n_examples)
            Sampled class indices.
        """
        n_classes = class_probs.shape[0]
        # each row: one MC repetition over all examples
        return self._rng.choice(
            n_classes,
            size=(self.num_samples, n_examples),
            p=class_probs
        )

    def _mc_conformal_quantile(self, scores, n_examples):
        """
        Compute the conformal quantile threshold from repeated scores.
        """
        # position of quantile
        q = (
            np.floor(self.alpha * self.num_samples * (n_examples + 1))
            - self.num_samples + 1
        ) / (n_examples * self.num_samples)
        # midpoint interpolation for quantile
        return np.quantile(scores, q, interpolation='midpoint')

    def _calibrate_threshold(self, conformity_scores, class_probs):
        """
        Estimate threshold by sampling labels and conformal scores.
        """
        n_examples = conformity_scores.shape[1]
        # sample MC labels
        labels_mc = self._sample_mc_labels(class_probs, n_examples)
        # repeat conformity scores per rep
        repeated = np.repeat(conformity_scores, self.num_samples, axis=0)
        # flatten label indices
        flat_labels = labels_mc.ravel()
        # pick matching conformity per draw
        idx = np.arange(flat_labels.shape[0])
        scores_flat = repeated[idx, flat_labels]
        # compute conformal quantile
        return self._mc_conformal_quantile(scores_flat, n_examples)

    def fit(self, X, y):
        """
        Fit selector by computing conformity (mutual information) and class distribution.
        """
        # number of features
        self.n_features = X.shape[1]
        # mutual information per feature
        mi = mutual_info_classif(X, y, random_state=self.random_state)
        self.conformity_scores = mi.reshape(1, -1)
        # empirical class probabilities
        classes, counts = np.unique(y, return_counts=True)
        self.class_probs = counts / float(len(y))
        # save fit data for later
        self.X_fitted = X
        self.feature_names = [f"Feature_{i}" for i in range(self.n_features)]
        return self

    def _compute_importance_scores(self, conformity, thresh):
        """
        Monte Carlo loop: add noise, threshold, average booleans.
        """
        scores = []
        for _ in range(self.num_samples):
            noise = self._rng.normal(loc=0.0,
                                     scale=0.1,
                                     size=conformity.shape)
            perturbed = conformity + noise
            mask = (perturbed > thresh).astype(float)
            scores.append(mask)
        arr = np.vstack(scores)
        # average over repetitions
        return np.mean(arr, axis=0)

    def transform(self, X, n_features_to_select):
        """
        Select top features from input X based on computed importance.
        """
        # calibrate threshold using MC conformal method
        self.threshold = self._calibrate_threshold(
            self.conformity_scores,
            self.class_probs
        )
        # compute final importance scores
        self.importance_scores = self._compute_importance_scores(
            self.conformity_scores,
            self.threshold
        )
        # pick top indices
        idx_sorted = np.argsort(self.importance_scores.flatten())
        chosen = idx_sorted[-n_features_to_select:]
        # return subset of X
        return X[:, chosen]

    def fit_transform(self, X, y, n_features_to_select):
        """
        Convenience: fit then transform in one call.
        """
        return self.fit(X, y).transform(X, n_features_to_select)
