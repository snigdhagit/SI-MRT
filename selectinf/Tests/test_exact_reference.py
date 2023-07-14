import numpy as np

from .instance import gaussian_instance
from ..lasso import lasso
from ..Utils.base import selected_targets

def test_inf(n=500,
             p=100,
             signal_fac=1.,
             s=5,
             sigma=2.,
             rho=0.4,
             randomizer_scale=1.,
             equicorrelated=False,
             CI=True):

    while True:

        inst, const = gaussian_instance, lasso.gaussian
        signal = np.sqrt(signal_fac * 2 * np.log(p))

        X, Y, beta = inst(n=n,
                          p=p,
                          signal=signal,
                          s=s,
                          equicorrelated=equicorrelated,
                          rho=rho,
                          sigma=sigma,
                          random_signs=True)[:3]

        n, p = X.shape

        sigma_ = np.std(Y)

        if n > (2 * p):
            dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n - p)
        else:
            dispersion = sigma_ ** 2

        eps = np.random.standard_normal((n, 2000)) * Y.std()
        W = 0.7 * np.median(np.abs(X.T.dot(eps)).max(1))

        conv = const(X,
                     Y,
                     W,
                     ridge_term=0.,
                     randomizer_scale=randomizer_scale * np.sqrt(dispersion))

        signs = conv.fit()
        nonzero = signs != 0
        print("size of selected set ", nonzero.sum())

        if nonzero.sum() > 0:
            beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))

            conv.setup_inference(dispersion=dispersion)

            target_spec = selected_targets(conv.loglike,
                                           conv.observed_soln,
                                           dispersion=dispersion)

            result_exact = conv.inference(target_spec)

            if CI is False:
                pvals = result_exact['pvalue']

                print("check pvalue ", pvals)
                return pvals

            else:
                intervals = np.asarray(result_exact[['lower_confidence', 'upper_confidence']])
                lci = intervals[:, 0]
                uci = intervals[:, 1]
                coverage = (lci < beta_target) * (uci > beta_target)
                length = uci - lci

                print("check intervals ", lci, uci)
                return np.mean(coverage), np.mean(length)


test_inf(CI=False)