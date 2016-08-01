import numpy as np

def predict_derivatives(self, x_new):
    """
    Predict derivatives of a gp at the points x_new.

    Author: Felix Berkenkamp (befelix)

    Parameters:
    -----------
    self: instance of GPy.core.gp
        With RBF kernel function
    x_new: 2d-array
        Every row is a new data point at which to evaluate the derivative

    Returns:
    --------
    mu: 2d-array
        The mean derivative
    var: 2/3d-array
        If there is only one data point to predict var is a 2d array with
        the variance matrix. Otherwise it is a 3d matrix where
        var[i, :, :] is the ith covariance matrix.
    """
    if not self.kern.name == 'rbf':
        raise AttributeError('The variance prior only works for RBF kernels.')
    x_new = np.atleast_2d(x_new)

    # Compute mean, initialize variance
    mu = self.kern.gradients_X(self.posterior.woodbury_vector.T, x_new, self.X)
    var = np.empty((x_new.shape[0], x_new.shape[1], x_new.shape[1]),
                   dtype=mu.dtype)

    # Make sure lengthscales are of the right dimensions
    lengthscales = self.kern.lengthscale.values
    if not lengthscales.shape[0] == self.X.shape[1]:
        lengthscales = np.tile(lengthscales, (self.X.shape[1],))

    def dk_dx(X, X2):
        """Compute the derivative of k(X, X2) with respect to X."""

        # Derivative with respect to r
        dK_dr = self.kern.dK_dr_via_X(X, X2)

        # Temporary stuff
        tmp = self.kern._inv_dist(X, X2) * dK_dr

        # dK_dX = dK_dr * dr_dx
        # dr_dx1 = invdist * (x1 - x'1) / l1**2
        dk_dx = np.empty((X2.shape[0], X.shape[1]), dtype=np.float64)
        for q in range(self.input_dim):
            dk_dx[:, q] = tmp * (X[:, q, None] - X2[None, :, q])
        return dk_dx / (lengthscales ** 2)

    # Compute derivative variance for each test point
    for i in range(x_new.shape[0]):
        dk = dk_dx(x_new[None, i, :], self.X)
        # Would be great if there was a way to get the prior directly from the
        # library
        # But I think only d1 is implemented
        var[i, :, :] = np.diag(self.kern.variance / (lengthscales ** 2)) -\
            np.dot(dk.T, np.dot(self.posterior.woodbury_inv, dk))

    # If there was only one test point to begin with, squeeze the
    # corresponding dimension
    if x_new.shape[0] <= 1:
        var = var.squeeze(0)
    return mu, var
