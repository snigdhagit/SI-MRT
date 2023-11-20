import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import norm, uniform
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import statsmodels.api as sm

#from .instance import gaussian_instance
from .instance_MRT import MRT_instance
from ..lasso import lasso
from ..Utils.base import selected_targets



def test_inf(n=500,
             p=100,
             signal_fac=1.,
             trueP = 5,
             s=5,
             sigma=2.,
             rho=0.4,
             randomizer_scale=1.,
             equicorrelated=False,
             CI=True):

    while True:

        inst, const = MRT_instance, lasso.WCLS
        #inst, const = MRT_instance, lasso.gaussian
        #signal = np.sqrt(signal_fac * 2 * np.log(p))

        # X, Y, beta = inst(n=n,
        #                   p=p,
        #                   signal=signal,
        #                   s=s,
        #                   equicorrelated=equicorrelated,
        #                   rho=rho,
        #                   sigma=sigma,
        #                   random_signs=True)[:3]


        X, Y, beta, A = MRT_instance(N=n, P=p, T=5, trueP = trueP)[:4]

        n, p = X.shape

        sigma_ = np.std(Y)

        if n > (2 * p):
            dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n - p)
        else:
            dispersion = sigma_ ** 2

        eps = np.random.standard_normal((n, 2000)) * Y.std()
        W = 0.7 * np.median(np.abs(X.T.dot(eps)).max(1))

        conv = const(A,
                     weights= np.ones(5),
                     feature_weights = W,
                     ridge_term=0.,
                     randomizer_scale=randomizer_scale * np.sqrt(dispersion))

        # conv = const(X,
        #              Y,
        #              W,
        #              ridge_term=0.,
        #              randomizer_scale=randomizer_scale * np.sqrt(dispersion))

        signs = conv.fit()
        nonzero = signs != 0
        #print("size of selected set ", nonzero.sum())

        if nonzero.sum() > 0:
            beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))

            conv.setup_inference(dispersion=dispersion)

            target_spec = selected_targets(conv.loglike,
                                           conv.observed_soln,
                                           dispersion=dispersion)

            result_exact = conv.inference(target_spec)

            if CI is False:
                pvals = result_exact['pvalue']
                #print("check pvalue ", pvals)
                return pvals

            else:
                intervals = np.asarray(result_exact[['lower_confidence', 'upper_confidence']])
                lci = intervals[:, 0]
                uci = intervals[:, 1]
                coverage = (lci < beta_target) * (uci > beta_target)
                length = uci - lci

                print("check intervals ", lci, uci)
                return np.mean(coverage), np.mean(length)

# print(test_inf(CI=True))

### Plots of Pivots/p-values

# nsim = 100
# mat = np.zeros((1,nusim))
# for i in range(nsim):
#     mat[:,i] = test_inf(CI=False)[:1]
#print(mat)

# y = mat[0]
# x = np.arange(1,101)
# plt.plot(x,y, color ="red")
# plt.show()

# qqplot(y,uniform,fit=True,line="45")
# plt.show()


# from statsmodels.distributions.empirical_distribution import ECDF
#
# nsim = 500
# pivots = []
# for i in range(nsim):
#     for j in test_inf(CI=False):
#         pivots.append(j)
#
# plt.clf()
# ecdf_pivot = ECDF(np.asarray(pivots))
# grid = np.linspace(0, 1, nsim)
# plt.plot(grid, ecdf_pivot(grid), c='blue', marker='^')
# plt.plot(grid, grid, 'k--')
# plt.show()



### Functions for Naive and Data Splitting


def compare_inf(n=500,
             p=100,
             signal_fac=1.,
             trueP = 5,
             s=5,
             sigma=2.,
             rho=0.4,
             randomizer_scale=1.,
             equicorrelated=False):

    while True:

        inst, const = MRT_instance, lasso.gaussian
        #inst, const = gaussian_instance, lasso.gaussian
        signal = np.sqrt(signal_fac * 2 * np.log(p))

        # X, Y, beta = inst(n=n,
        #                   p=p,
        #                   signal=signal,
        #                   s=s,
        #                   equicorrelated=equicorrelated,
        #                   rho=rho,
        #                   sigma=sigma,
        #                   random_signs=True)[:3]


        X, Y, beta = MRT_instance(N=n, P=p, T=1, trueP = trueP)[:3]

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
        #print("size of selected set ", nonzero.sum())

        #Sel-Inf
        if nonzero.sum() > 0:
            beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))

            conv.setup_inference(dispersion=dispersion)

            target_spec = selected_targets(conv.loglike,
                                           conv.observed_soln,
                                           dispersion=dispersion)

            result_exact = conv.inference(target_spec)
            sel_intervals = np.asarray(result_exact[['lower_confidence', 'upper_confidence']])
            lci = sel_intervals[:, 0]
            uci = sel_intervals[:, 1]
            coverage1 = (lci < beta_target) * (uci > beta_target)
            length1 = uci - lci
            #print("Sel Inf coverage and length ", np.mean(coverage1), np.mean(length1))
            #print("check selective intervals ", lci, uci)

        #Naive
        olsn = sm.OLS(Y, X[:, nonzero])
        olsfit = olsn.fit()
        naive_intervals = olsfit.conf_int(alpha=0.1, cols=None)
        lci = naive_intervals[:, 0]
        uci = naive_intervals[:, 1]
        coverage2 = (lci < beta_target) * (uci > beta_target)
        length2 = uci - lci
        #print("Naive coverage and length ", np.mean(coverage2), np.mean(length2))

        #Split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 100, random_state = 420)
        eps1 = np.random.standard_normal((200, 2000)) * Y_train.std()
        W_train = 0.7 * np.median(np.abs(X_train.T.dot(eps1)).max(1))
        conv2 = const(X_train,
                      Y_train,
                      W_train,
                      ridge_term=0.,
                      randomizer_scale=randomizer_scale * np.sqrt(dispersion))

        signs2 = conv2.fit()
        nonzero2 = signs2 != 0
        beta_target2 = np.linalg.pinv(X_test[:, nonzero2]).dot(X_test.dot(beta))



        olss = sm.OLS(Y_test, X_test[:, nonzero2])
        olsfit2 = olss.fit()
        split_intervals = olsfit2.conf_int(alpha=0.1, cols=None)
        lci = split_intervals[:, 0]
        uci = split_intervals[:, 1]
        coverage3 = (lci < beta_target2) * (uci > beta_target2)
        length3 = uci - lci
        # print("size of selected set ", nonzero.sum())
        #print("Split coverage and length ", np.mean(coverage3), np.mean(length3))

        return coverage1, coverage2, coverage3, length1, length2, length3

#print(compare_inf(n=500))

# nsim = 500
# bcoverage1 = []
# bcoverage2 = []
# bcoverage3 = []
# blength1 = []
# blength2 = []
# blength3 = []
# for i in range(nsim):
#     coverage1, coverage2, coverage3, length1, length2, length3 = compare_inf(n=500)
#     bcoverage1.append(np.mean(coverage1))
#     bcoverage2.append(np.mean(coverage2))
#     bcoverage3.append(np.mean(coverage3))
#     blength1.append(np.mean(length1))
#     blength2.append(np.mean(length2))
#     blength3.append(np.mean(length3))
#
# Coverage_Data = pd.DataFrame({"Selective Coverage": bcoverage1, "Naive Coverage": bcoverage2, "Data Splitting Coverage": bcoverage3})
# Length_Data = pd.DataFrame({"Selective CI Lengths": blength1, "Naive CI Lengths": blength2, "Data Splitting CI Lengths": blength3})
#
# print(Coverage_Data)
# print(Length_Data)
#
# Coverage_Data.to_csv('Coverage_Data.csv')
# Length_Data.to_csv('Length_Data.csv')
# print(np.mean(np.asarray(bcoverage1)))
# print(np.mean(np.asarray(bcoverage2)))
# print(np.mean(np.asarray(bcoverage3)))
# print(np.mean(np.asarray(blength1)))
# print(np.mean(np.asarray(blength2)))
# print(np.mean(np.asarray(blength3)))

# Comparison Plots
# Set the figure size
# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True
# plt.style.use('seaborn')
#
# # Plot the dataframe
# plot_cov = Coverage_Data[['Selective Coverage', 'Naive Coverage', 'Data Splitting Coverage']].plot(kind='box', title='Coverage')
# plt.show()
# plot_length = Length_Data[['Selective CI Lengths', 'Data Splitting CI Lengths']].plot(kind='box', title='CI Length')
# plt.show()


# ON REAL DATA

# X = np.asarray(pd.read_csv(r'~/Documents/Xoutput.csv'))


# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit(X)
# X = scaler.transform(X)

# np.c_[np.ones(15641), X]

# names = np.asarray(pd.read_csv(r'~/Documents/names.csv'))
# Y = np.asarray(pd.read_csv(r'~/Documents/Youtput.csv'))
# Y = Y.reshape((15641,))
# n, p = X.shape

# sigma_ = np.std(Y)
#
# if n > (2 * p):
#     dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n - p)
# else:
#     dispersion = sigma_ ** 2
#
# eps = np.random.standard_normal((n, 2000)) * Y.std()
# W = 50 * np.median(np.abs(X.T.dot(eps)).max(1))
#
# const = lasso.gaussian
# randomizer_scale = 1.
# conv = const(X,
#              Y,
#              W,
#              ridge_term=0.,
#              randomizer_scale=randomizer_scale * np.sqrt(dispersion))
#
# signs = conv.fit()
# nonzero = signs != 0
# print("size of selected set ", nonzero.sum())
# print("potential moderators ", names[nonzero])
#
# conv.setup_inference(dispersion=dispersion)
#
# target_spec = selected_targets(conv.loglike,
#                                conv.observed_soln,
#                                dispersion=dispersion)
#
# result_exact = conv.inference(target_spec)
#
# intervals = np.asarray(result_exact[['lower_confidence', 'upper_confidence']])
# lci = intervals[:, 0]
# uci = intervals[:, 1]
# print("check intervals ", lci, uci)


#means and sds of selected columns

# print(np.mean(X[:,nonzero], axis=0))
# print(np.std(X[:,nonzero], axis=0))



nsim = 10
coverage1 = []

for i in range(nsim):
    coverage1 = test_inf(CI=True)[0]

print(np.mean(coverage1))

