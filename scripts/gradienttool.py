from __future__ import absolute_import, division, print_function

import numpy as np
import scipy as sp
import scipy.stats as sps
import shared as ip2
import matplotlib.pyplot as plt

import pandas as pd

import GPy
import GPy.plotting.Tango as Tango

def zerosLinearInterpolate(xy):
    """Linearly interpolate to find all zeros from an evaluated function"""
    ii = np.where((np.diff(xy[:,1] < 0)))[0]
    out = np.empty(len(ii),dtype=xy.dtype)
    for j,i in enumerate(ii):
        x0,y0 = xy[i+0,[0,1]]
        x1,y1 = xy[i+1,[0,1]]

        m = (y0 - y1) / (x0 - x1)
        c = y1 - m*x1
        out[j] = -c/m
    return out

def _test():
    assert np.allclose(0.5,zerosLinearInterpolate(np.array([[0,-0.5],
                                                            [1, 0.5]])))

def doReportPlot(fig, g, title=None,
                 CI=sps.norm.ppf([0.025,0.975])):
    """Generate a pair of plots for the report"""
    res1 = g.getResults()
    ar = ip2.niceAxisRange(res1.index)
    res2 = g.getResults(np.linspace(ar[0],ar[1],101))

    # extract for ease of access
    mu     = res2['mu']
    var    = res2['var']
    mud    = res2['mud']
    tscore = res1['tscore']

    # set up plots
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, sharex=ax1)

    if title is not None: ax1.set_title(title)

    ax1.margins(0.04)
    ax2.set_ylim([-6,6])

    # draw data, function and error bars
    mui = np.array(mu.index, dtype=float) # needed for Python2.7
    ax1.fill_between(mui,
                     mu + CI[0]*np.sqrt(var + g.noiseVariance),
                     mu + CI[1]*np.sqrt(var + g.noiseVariance),
                     facecolor='black', edgecolor='none', alpha=0.05)
    ax1.fill_between(mui,
                     mu + CI[0]*np.sqrt(var),
                     mu + CI[1]*np.sqrt(var),
                     facecolor='royalblue', edgecolor='none', alpha=0.2)
    mu.plot(ax=ax1)
    # data goes at the end so it doesn't get hidden by everything else
    ax1.scatter(g.X, g.Y, marker='x', c='black', s=40, lw=1.2);

    # draw t-scores
    ax2.vlines(tscore.index, 0, tscore)
    ax2.plot(res2.index, res2['tscore'])
    for y in CI:
        ax2.axhline(y, lw=1, ls='--', color='black')

class GradientTool(object):
    """Code for running WSBC's Gradient-Tool analysis

    :param X: time points
    :param Y: the readings values
    """

    def __init__(self, X, Y):
        assert len(X.shape) == 1
        assert X.shape == Y.shape
        self._X = X
        self._Y = Y
        self.m = GPy.models.GPRegression(X[:,None], Y[:,None])

    @property
    def X(self): return self._X
    @property
    def Y(self): return self._Y

    @property
    def rbfLengthscale(self): return float(self.m.rbf.lengthscale)
    @property
    def rbfVariance   (self): return float(self.m.rbf.variance)
    @property
    def noiseVariance (self): return float(self.m.Gaussian_noise.variance)

    def setPriorRbfLengthscale(self, shape, rate):
        self.m.rbf.lengthscale.set_prior(GPy.priors.Gamma(shape, rate), warning=False)
    def setPriorRbfVariance(self, shape, rate):
        self.m.rbf.variance.set_prior(GPy.priors.Gamma(shape, rate), warning=False)
    def setPriorNoiseVariance(self, shape, rate):
        self.m.Gaussian_noise.variance.set_prior(GPy.priors.Gamma(shape, rate), warning=False)

    def optimize(self):
        self.m.optimize()

    def getResults(self, Xstar=None):
        """Returns a Pandas DataFrame of the latent function, its derivative and variances.

        The `index` is time"""

        # get the points we're evaluating the function at
        if Xstar is None:
            Xstar = np.unique(self._X)

        # get predictions of our function and its derivative
        mu,var = self.m._raw_predict(Xstar[:,None])
        mud,vard = ip2.predict_derivatives(self.m, Xstar[:,None])

        # put into a matrix
        out = np.empty((len(Xstar),5),dtype=self.Y.dtype)
        out[:,0] = mu[:,0]
        out[:,1] = var[:,0]
        out[:,2] = mud[:,0]
        out[:,3] = vard[:,0,0]
        out[:,4] = out[:,2]/np.sqrt(out[:,3])

        # convert to a DataFrame and return
        return pd.DataFrame(out, index=Xstar, columns=
                            ["mu","var","mud","vard","tscore"])

class GradientToolNormalised(GradientTool):
    """Wrap the basic GradientTool up such that X and Y are both """
    def __init__(self, X, Y, xtrans=None, ytrans=None):
        if xtrans is not None: xtrans = (min(X), max(X)-min(X))
        if ytrans is not None: ytrans = (np.mean(Y), np.std(Y))

        self.tx = xtrans
        self.ty = ytrans

        super(GradientToolNormalised, self).__init__((X - self.tx[0]) / self.tx[1],
                                                     (Y - self.ty[0]) / self.ty[1])

    @property
    def X(self): return self._X * self.tx[1] + self.tx[0]
    @property
    def Y(self): return self._Y * self.ty[1] + self.ty[0]

    @property
    def rbfLengthscale(self): return float(self.m.rbf.lengthscale) * self.tx[1]**2
    @property
    def rbfVariance   (self): return float(self.m.rbf.variance) * self.ty[1]**2
    @property
    def noiseVariance (self): return float(self.m.Gaussian_noise.variance) * self.ty[1]**2

    def getResults(self, Xstar=None):
        if Xstar is not None:
            Xstar = (Xstar - self.tx[0]) / self.tx[1]

        res = super(GradientToolNormalised, self).getResults(Xstar)

        res.set_index(res.index * self.tx[1] + self.tx[0], inplace=True)

        res["mu"]   = res["mu"] * self.ty[1] + self.ty[0]
        res["var"]  = res["var"] * self.ty[1] ** 2
        res["mud"]  = res["mud"] * (self.ty[1]/self.tx[1])
        res["vard"] = res["vard"] * (self.ty[1]/self.tx[1])**2

        return res
