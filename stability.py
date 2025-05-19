import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.utils import check_random_state
from select import Selector


class GlobalRelevanceEvaluator:
    """
    Compute and rank global feature relevance from a fitted Selector.
    """

    def __init__(self, model: Selector):
        self._model = model

    def evaluate(self) -> pd.DataFrame:
        """
        Build a DataFrame of features with their relevance scores,
        sorted descending by score.
        """
        scores = self._model.importance_scores.flatten()
        labels = self._model.feature_names
        df = pd.DataFrame({
            'feature_label': labels,
            'relevance_value': scores
        })
        return df.sort_values('relevance_value', ascending=False)

    def pick_top(self, count: int) -> list:
        """
        Return the names of the top `count` features by relevance.
        """
        flat_scores = np.ravel(self._model.importance_scores)
        top_idxs = np.argsort(flat_scores)[-count:]
        return [self._model.feature_names[i] for i in top_idxs]

    def extract_cutoff(self) -> float:
        """
        Retrieve the numeric cutoff used internally by Selector.
        """
        return self._model.threshold


class ConsistencyAnalyzer:
    """
    Measure how stable feature relevance is across random subsamples.
    """

    def __init__(self, prototype: Selector):
        self._prototype = prototype

    def assess(self,
               features: np.ndarray,
               targets: np.ndarray,
               repeats: int = 10,
               fraction: float = 0.8) -> pd.DataFrame:
        """
        Fit Selector on random subsamples and average scores to gauge stability.
        """
        n_samples, n_features = features.shape
        subsample_n = int(n_samples * fraction)
        stability_mat = np.zeros((repeats, n_features))

        for _ in range(repeats):
            idxs = np.random.choice(n_samples, subsample_n, replace=False)
            sub_X, sub_y = features[idxs], targets[idxs]

            clone = Selector(alpha=self._prototype.alpha,
                                 num_samples=self._prototype.num_samples,
                                 random_state=self._prototype.random_state)
            clone.fit(sub_X, sub_y)
            _ = clone.transform(sub_X, self._prototype.n_features)

            stability_mat[_] = clone.importance_scores.flatten()

        avg_values = np.mean(stability_mat, axis=0)
        return pd.DataFrame({
            'feature_label': self._prototype.feature_names,
            'stability_value': avg_values
        }).sort_values('stability_value', ascending=False)


class LocalRelevanceEstimator:
    """
    Estimate feature importance around one specific instance via perturbation.
    """

    def __init__(self, trained: Selector):
        self._trained = trained

    def estimate(self,
                 instance: np.ndarray,
                 n_draws: int = 1000,
                 kernel_bw: float = 0.75) -> pd.DataFrame:
        """
        Generate perturbed points around `instance`, weight them by distance,
        and average importance draws using numpy RNG.
        """
        if not hasattr(self._trained, 'X_fitted'):
            raise ValueError("Selector must be fitted before local estimation.")

        base = np.array(instance).reshape(1, -1)
        noise_scale = np.std(self._trained.X_fitted, axis=0) * 0.1
        draws = np.random.normal(loc=base,
                                 scale=noise_scale,
                                 size=(n_draws, self._trained.n_features))

        dists = cdist(base, draws, metric='euclidean').flatten()
        weights = np.exp(-(dists ** 2) / (kernel_bw ** 2))

        rng = check_random_state(self._trained.random_state)
        local_scores = []
        for _ in range(n_draws):
            seed = rng.randint(0, 2**32)
            # _compute_importance_scores now expects a numpy RNG seed
            score = self._trained._compute_importance_scores(
                self._trained.conformity_scores,
                self._trained.threshold,
                seed
            )
            local_scores.append(score.flatten())

        arr_scores = np.array(local_scores)
        weighted = np.average(arr_scores, axis=0, weights=weights).flatten()

        return pd.DataFrame({
            'feature_label': self._trained.feature_names,
            'local_relevance': weighted
        }).sort_values('local_relevance', ascending=False)


class BarChartDrawer:
    """
    Plot any two‐column DataFrame as a horizontal bar chart.
    """

    def render(self,
               df: pd.DataFrame,
               val_key: str,
               lbl_key: str,
               title: str,
               top_n: int = 10):
        snippet = df.head(top_n)
        plt.figure(figsize=(12, 8))
        sns.barplot(x=val_key, y=lbl_key, data=snippet)
        plt.title(title)
        plt.tight_layout()
        plt.show()


class InterpretationManager:
    """
    Facade to run global, stability, and local analyses + plotting.
    """

    def __init__(self, engine: Selector):
        self._engine = engine
        self._global = GlobalRelevanceEvaluator(engine)
        self._stability = ConsistencyAnalyzer(engine)
        self._local = LocalRelevanceEstimator(engine)
        self._drawer = BarChartDrawer()

    def global_relevance(self) -> pd.DataFrame:
        return self._global.evaluate()

    def show_global(self, top_n: int = 10):
        df = self._global.evaluate()
        self._drawer.render(df,
                            val_key='relevance_value',
                            lbl_key='feature_label',
                            title=f'Top {top_n} Global Features',
                            top_n=top_n)

    def stability_profile(self,
                          X: np.ndarray,
                          y: np.ndarray,
                          repeats: int = 10,
                          fraction: float = 0.8) -> pd.DataFrame:
        return self._stability.assess(X, y, repeats, fraction)

    def show_stability(self,
                       stab_df: pd.DataFrame,
                       top_n: int = 10):
        self._drawer.render(stab_df,
                            val_key='stability_value',
                            lbl_key='feature_label',
                            title=f'Top {top_n} Stable Features',
                            top_n=top_n)

    def local_relevance(self,
                        sample: np.ndarray,
                        draws: int = 1000,
                        bw: float = 0.75) -> pd.DataFrame:
        return self._local.estimate(sample, draws, bw)

    def show_local(self,
                   loc_df: pd.DataFrame,
                   top_n: int = 10):
        self._drawer.render(loc_df,
                            val_key='local_relevance',
                            lbl_key='feature_label',
                            title=f'Top {top_n} Local Features',
                            top_n=top_n)
