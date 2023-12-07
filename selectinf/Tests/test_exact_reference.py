import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import norm, uniform
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import statsmodels.api as sm

#from .instance import gaussian_instance
from .MRT_instance import MRT_instance
from ..lasso import lasso
from ..Utils.base import selected_targets, selected_targets_WCLS




def test_inf(N=300,
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

        # inst, const = MRT_instance, lasso.WCLS
        inst, const = MRT_instance, lasso.gaussian
        #signal = np.sqrt(signal_fac * 2 * np.log(p))

        # X, Y, beta = inst(n=n,
        #                   p=p,
        #                   signal=signal,
        #                   s=s,
        #                   equicorrelated=equicorrelated,
        #                   rho=rho,
        #                   sigma=sigma,
        #                   random_signs=True)[:3]


        X, Y, beta, A = MRT_instance(N=N)[:4]

        n1, p = X.shape

        n = int(n1 / 5)


        sigma_ = np.std(Y)

        if n > (2 * p):
            dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n1 - p)
        else:
            dispersion = sigma_ ** 2

        eps = np.random.standard_normal((n1, 2000)) * Y.std()
        W = 0.7 * np.median(np.abs(X.T.dot(eps)).max(1))

        # conv = const(A,
        #              weights= np.ones(5),
        #              feature_weights = W,
        #              ridge_term=0.,
        #              randomizer_scale=randomizer_scale * np.sqrt(dispersion))

        conv = const(X,
                     Y,
                     W,
                     ridge_term=0.,
                     randomizer_scale=randomizer_scale * np.sqrt(dispersion))

        signs = conv.fit()
        nonzero = signs != 0
        #print("size of selected set ", nonzero.sum())

        if nonzero.sum() > 0:
            beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))

            conv.setup_inference(dispersion=dispersion)

            # target_spec = selected_targets(conv.loglike,
            #                                conv.observed_soln,
            #                                dispersion=dispersion)


            target_spec = selected_targets_WCLS(conv.loglike,
                                           conv.observed_soln,
                                           K = conv.K,
                                           dispersion= 1)

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

                # print("check intervals ", lci, uci)
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


def compare_inf(N=300,
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
        # signal = np.sqrt(signal_fac * 2 * np.log(p))

        # X, Y, beta = inst(n=n,
        #                   p=p,
        #                   signal=signal,
        #                   s=s,
        #                   equicorrelated=equicorrelated,
        #                   rho=rho,
        #                   sigma=sigma,
        #                   random_signs=True)[:3]


        X, Y, beta, A = MRT_instance(N=N)[:4]

        n1, p = X.shape

        n = int(n1 / 30)

        sigma_ = np.std(Y)

        if n > (2 * p):
            dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n1 - p)
        else:
            dispersion = sigma_ ** 2

        eps = np.random.standard_normal((n1, 2000)) * Y.std()
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

            target_spec2 = selected_targets(conv.loglike,
                                           conv.observed_soln,
                                           dispersion=dispersion)

            target_spec = selected_targets_WCLS(conv.loglike,
                                                A,
                                                conv.observed_soln,
                                                K = conv.K,
                                                dispersion= 1)

            result_exact = conv.inference(target_spec)
            sel_intervals = np.asarray(result_exact[['lower_confidence', 'upper_confidence']])
            lci = sel_intervals[:, 0]
            uci = sel_intervals[:, 1]
            coverage1 = (lci < beta_target) * (uci > beta_target)
            length1 = uci - lci
            #print("Sel Inf coverage and length ", np.mean(coverage1), np.mean(length1))
            #print("check selective intervals ", lci, uci)

            # without sandwiched cov
            result_exact2 = conv.inference(target_spec2)
            sel_intervals2 = np.asarray(result_exact2[['lower_confidence', 'upper_confidence']])
            lci2 = sel_intervals2[:, 0]
            uci2 = sel_intervals2[:, 1]
            coverage1w = (lci2 < beta_target) * (uci2 > beta_target)
            length1w = uci2 - lci2

        #Naive

        # olsn = sm.OLS(Y, X[:, nonzero])
        # olsfit = olsn.fit()
        # naive_intervals = olsfit.conf_int(alpha=0.1, cols=None)
        # lci = naive_intervals[:, 0]
        # uci = naive_intervals[:, 1]
        # coverage2 = (lci < beta_target) * (uci > beta_target)
        # length2 = uci - lci
        #print("Naive coverage and length ", np.mean(coverage2), np.mean(length2))

        Xf = np.array(A.iloc[:, 2:-1])
        Xf = Xf[:, nonzero]
        yf = A['Y']
        groups = A['id']

        # fit the GEE model
        model = sm.GEE(yf, Xf, groups=groups, family=sm.families.Gaussian(), cov_struct=sm.cov_struct.Independence())
        result = model.fit(cov_type='robust')

        GEE_intervals = np.array(result.conf_int(alpha=0.1))
        lci = GEE_intervals[:, 0]
        uci = GEE_intervals[:, 1]
        coverage2 = (lci < beta_target) * (uci > beta_target)
        length2 = uci - lci



        #Split
        unique_ids = A['id'].unique()
        shuffled_ids = pd.Series(unique_ids).sample(frac=1, random_state=42019).tolist()

        # Split the IDs into train and test sets
        test_size = 0.3
        train_size = int((1 - test_size) * len(shuffled_ids))
        train_ids = shuffled_ids[:train_size]
        test_ids = shuffled_ids[train_size:]

        # Split the dataframe based on these IDs
        A_train = A[A['id'].isin(train_ids)]
        A_test = A[A['id'].isin(test_ids)]
        X_train = np.array(A_train.iloc[:, 2:p+2])
        X_test = np.array(A_test.iloc[:, 2:p+2])
        Y_train = np.array(A_train.iloc[:, p + 2])
        Y_test = np.array(A_test.iloc[:, p + 2])

        n_train = X_train.shape[0]

        eps1 = np.random.standard_normal((n_train, 2000)) * Y_train.std()
        W_train = 0.7 * np.median(np.abs(X_train.T.dot(eps1)).max(1))
        conv2 = const(X_train,
                      Y_train,
                      W_train,
                      ridge_term=0.,
                      randomizer_scale=randomizer_scale * np.sqrt(dispersion))

        signs2 = conv2.fit()
        nonzero2 = signs2 != 0
        beta_target2 = np.linalg.pinv(X_test[:, nonzero2]).dot(X_test.dot(beta))


        Xf1 = np.array(A_test.iloc[:, 2:-1])
        Xf1 = Xf1[:, nonzero2]
        yf1 = A_test['Y']
        groups = A_test['id']

        # fit the GEE model
        model2 = sm.GEE(yf1, Xf1, groups=groups, family=sm.families.Gaussian(), cov_struct=sm.cov_struct.Independence())
        result_split = model2.fit(cov_type='robust')

        split_intervals = np.array(result_split.conf_int(alpha=0.1))

        # olss = sm.OLS(Y_test, X_test[:, nonzero2])
        # olsfit2 = olss.fit()
        # split_intervals = olsfit2.conf_int(alpha=0.1, cols=None)

        lci = split_intervals[:, 0]
        uci = split_intervals[:, 1]
        coverage3 = (lci < beta_target2) * (uci > beta_target2)
        length3 = uci - lci
        # print("size of selected set ", nonzero.sum())
        #print("Split coverage and length ", np.mean(coverage3), np.mean(length3))

        return coverage1, coverage1w, coverage2, coverage3, length1, length1w, length2, length3

# print(compare_inf(300))

nsim = 500
bcoverage1 = []
bcoverage1w = []
bcoverage2 = []
bcoverage3 = []
blength1 = []
blength1w = []
blength2 = []
blength3 = []
for i in range(nsim):
    coverage1, coverage1w, coverage2, coverage3, length1, length1w, length2, length3 = compare_inf(3000)
    bcoverage1.append(np.mean(coverage1))
    bcoverage1w.append(np.mean(coverage1w))
    bcoverage2.append(np.mean(coverage2))
    bcoverage3.append(np.mean(coverage3))
    blength1.append(np.mean(length1))
    blength1w.append(np.mean(length1w))
    blength2.append(np.mean(length2))
    blength3.append(np.mean(length3))

Coverage_Data = pd.DataFrame({"Selective Coverage": bcoverage1, "Selective Coverage (NS)": bcoverage1w, "Naive Coverage": bcoverage2, "Data Splitting Coverage": bcoverage3})
Length_Data = pd.DataFrame({"Selective CI Lengths": blength1, "Selective CI Lengths (NS)": blength1w, "Naive CI Lengths": blength2, "Data Splitting CI Lengths": blength3})

# print(Coverage_Data)
# print(Length_Data)
#
# Coverage_Data.to_csv('Coverage_Data.csv')
# Length_Data.to_csv('Length_Data.csv')
print(np.mean(np.asarray(bcoverage1)))
print(np.mean(np.asarray(bcoverage1w)))
print(np.mean(np.asarray(bcoverage2)))
print(np.mean(np.asarray(bcoverage3)))
print(np.mean(np.asarray(blength1)))
print(np.mean(np.asarray(blength1w)))
print(np.mean(np.asarray(blength2)))
print(np.mean(np.asarray(blength3)))

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



# nsim = 500
# coverage1 = []
# length1 = []
#
# for i in range(nsim):
#     coverage1.append(test_inf(CI=True)[0])
#     length1.append(test_inf(CI=True)[1])
#
# print(np.mean(coverage1))
# print(np.mean(length1))


