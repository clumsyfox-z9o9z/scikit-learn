import numpy as np

from .base import BaseMixture


class BernoulliMixture(BaseMixture):

    def __init__(self, n_components=1, tol=1e-3, reg_covar=1e-6,
                 max_iter=100, n_init=1, init_params='kmeans',
                 weights_init=None, means_init=None, precisions_init=None,
                 random_state=None, warm_start=False,
                 verbose=0, verbose_interval=10):
        super(BernoulliMixture).__init__(
            n_components=n_components, tol=tol, reg_covar=reg_covar,
            max_iter=max_iter, n_init=n_init, init_params=init_params,
            random_state=random_state, warm_start=warm_start,
            verbose=verbose, verbose_interval=verbose_interval)


    def _check_parameters(self, X):
        """Check initial parameters of the derived class.

        Parameters
        ----------
        X : array-like, shape  (n_samples, n_features)
        """
        pass

    def _initialize(self, X, resp):
        """Initialize the model parameters of the derived class.

        Parameters
        ----------
        X : array-like, shape  (n_samples, n_features)

        resp : array-like, shape (n_samples, n_components)
        """
        pass

    def _m_step(self, X, log_resp):
        """M step.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        log_resp : array-like, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        n_samples, _ = X.shape
        self.weights_, self.means_ = (_estimate_bernoulli_parameters(X, np.exp(log_resp)))
        self.weights_ /= n_samples

    def _check_is_fitted(self):
        pass

    def _get_parameters(self):
        pass

    def _set_parameters(self, params):
        pass

    def _estimate_log_weights(self):
        """Estimate log-weights in EM algorithm, E[ log pi ] in VB algorithm.

        Returns
        -------
        log_weight : array, shape (n_components, )
        """
        pass

    def _estimate_log_prob(self, X):
        """Estimate the log-probabilities log P(X | Z).

        Compute the log-probabilities per each component for each sample.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        log_prob : array, shape (n_samples, n_component)
        """
        pass