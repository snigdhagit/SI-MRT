import numpy as np
import pandas as pd
from statsmodels.tsa.arima_process import ArmaProcess
import statsmodels.api as sm

# Define constants
T = 30
N = 900
P = 50
trueP = 3
sigma_residual = 1
sigma_randint = 1.5
rho = 0.7
txt_intercept = -0.2
beta_logit = np.concatenate(([-1], 0.8 * np.ones(P) / P))
beta_11 = 2
theta1 = 0.8

# Generate AR(1) process
def arima_sim(rho, n, sd=1):
    ar = np.array([1, -rho])
    ma = np.array([1])
    ARMA = ArmaProcess(ar, ma)
    return ARMA.generate_sample(nsample=n, scale=sd)


# Generate individual data
def generate_individual(id=1):
    all_states = np.column_stack([arima_sim(rho, T) for _ in range(P)])

    all_actions = np.zeros(T)
    all_probabilities = np.zeros(T)

    current_action = 0
    for t in range(T):
        current_states = all_states[t, :]
        current_action = all_actions[t - 1] if t != 0 else 0
        prob_action = 1 / (1 + np.exp(-np.dot(np.concatenate(([current_action], current_states)), beta_logit)))
        current_action = np.random.binomial(n=1, p=prob_action)
        all_probabilities[t] = prob_action
        all_actions[t] = current_action

    treatment_effect = np.sum(all_states[:, :trueP], axis=1) * beta_11 / trueP + txt_intercept
    main_effect = theta1 * np.sum(all_states, axis=1)
    meanY = main_effect + (all_actions - all_probabilities) * treatment_effect
    errorY = arima_sim(rho, T)
    txterrorY = arima_sim(rho, T, sd=sigma_residual) + np.random.normal(0, sigma_randint, T)
    obsY = meanY + errorY + txterrorY * (all_actions - all_probabilities)

    df_individual = pd.DataFrame({
        "id": id,
        "decision_point": np.arange(1, T + 1),
        **{f"state{i}": all_states[:, i] for i in range(P)},
        "prob": all_probabilities,
        "action": all_actions,
        "outcome": obsY
    })
    return df_individual

def MRT_instance(N=N):

    individual_data_frames = []

    # Generate individual data and collect them in the list
    for n in range(1, N + 1):
          fake_individual = generate_individual(n)  # Call the generate_individual function
          individual_data_frames.append(fake_individual)

    MRT_data = pd.concat(individual_data_frames, ignore_index=True)
    n1 = int(2 * N / 3)
    X1 = MRT_data[MRT_data["id"] > n1].iloc[:, 2:P + 2]
    Y1 = MRT_data[MRT_data["id"] > n1].iloc[:, P + 4]

    # alphahat = np.array(sm.OLS(Y1, X1).fit().params)
    alphahat = theta1 * np.ones(P)
    Y = np.array(MRT_data[MRT_data["id"] < n1 + 1].iloc[:, P + 4]) - np.dot(np.array(MRT_data[MRT_data["id"] < n1 + 1].iloc[:, 2:P + 2]), alphahat)
    At_Pt = np.array(MRT_data[MRT_data["id"] < n1 + 1].iloc[:, P + 3]) - np.array(MRT_data[MRT_data["id"] < n1 + 1].iloc[:, P + 2])
    X = np.array(MRT_data[MRT_data["id"] < n1 + 1].iloc[:, 2:P + 2].multiply(At_Pt, axis="index"))
    # X -= X.mean(0)[None, :] #centering
    scaling = X.std(0) * np.sqrt(N)
    # X /= np.sqrt(N) #scaling
    
    beta = (beta_11/trueP) * np.concatenate((np.ones(trueP), np.zeros(P - trueP)))
    A = MRT_data[MRT_data["id"] < n1 + 1].iloc[:, :2]
    A = A.join(pd.DataFrame(X, columns = ['State'+str(i) for i in range(1,P+1)]))
    A['Y'] = Y.tolist()


    # active = np.zeros(P, bool)
    # active[beta != 0] = True

    # scaling = Y.std(0) * np.sqrt(n)
    # Y /= scaling
    return X, Y, beta, A


# X, Y, beta, MRT = MRT_instance(N=300)[:4]
# print(MRT)
# print(MRT.shape)
# print(X.shape)
# print(MRT.columns)
# print(max(MRT.id))
# MRT.to_csv('MRT.csv')

