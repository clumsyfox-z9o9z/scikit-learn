import numpy as np

from .base import BaseMixture


def _estimate_bernoulli_parameters(X, resp):
    """Estimate the Bernoulli distribution parameters.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The input data array.

    resp : array-like, shape (n_samples, n_components)
        The responsibilities for each data sample in X.

    Returns
    -------
    nk : array-like, shape (n_components,)
        The numbers of data samples in the current components.

    means : array-like, shape (n_components, n_features)
        The centers of the current components.
    """
    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    means = np.dot(resp.T, X) / nk[:, np.newaxis]
    return nk, means


def _estimate_log_bernoulli_prob(X, means):
    """Estimate the log Bernoulli probability

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)

    means : array-like, shape (n_components, n_features)

    Returns
    -------
    log_prob : array, shape (n_samples, n_components)
    """
    n_samples, n_features = X.shape
    n_components, _ = means.shape

    log_prob = np.empty((n_samples, n_components))

    for k in range(n_components):
        for n in range(n_samples):
            y = X[n, :] * np.log(means[k, :]) + (1 - X[n, :]) * np.log(1 - means[k, :])
            log_prob[n, k] = np.sum(y, axis=1)

    return log_prob


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
        return (self.weights_, self.means_)

    def _set_parameters(self, params):
        (self.weights_, self.means_) = params

    def _estimate_log_weights(self):
        return np.log(self.weights_)

    def _estimate_log_prob(self, X):
        return _estimate_log_bernoulli_prob(X, self.means_)
