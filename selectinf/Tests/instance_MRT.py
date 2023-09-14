import numpy as np
import pandas as pd
import statsmodels.api as sm


T = 1 # Number of timepoints
N = 500 # Number of individuals
P = 100 # Number of potential moderators
trueP = 5 # Number of true moderators

### PARAMETER VALUES
sigma_residual = 0.5
sigma_AR = 0.5
signal_strength = 0.2
baseline_strength = -0.2
weight = 0.75
alpha = 0.9

beta_11 = 0.1
theta1 = 0.9
#Gamma = np.full((P, P), alpha * (1 - weight) / (P - 1))
#np.fill_diagonal(Gamma,weight*alpha)
#Var = np.power(sigma_AR, np.abs(np.subtract.outer(np.arange(1,T+1), np.arange(1,T+1))))
#C = np.linalg.cholesky(Var)

def generate_individual(id,
                        T=T,
                        P=P,
                        trueP=trueP,
                        alpha=0.9,
                        sigma_residual=0.5,
                        sigma_AR=0.7,
                        beta_11=0.4,
                        theta1=0.8,
                        signal_strength=0.2,
                        baseline_strength=-0.5):


    beta_logit = np.hstack((-1, 1.6 * np.ones(P) / P))
    Gamma = np.full((P, P), alpha * (1 - weight) / (P - 1))
    np.fill_diagonal(Gamma, weight * alpha)
    Var = np.power(sigma_AR, np.abs(np.subtract.outer(np.arange(1, T + 1), np.arange(1, T + 1))))
    C = np.linalg.cholesky(Var)
    all_states = np.zeros((T, P))
    all_states_mean = np.zeros((T, P))
    all_actions = np.zeros(T)
    all_probabilities = np.zeros(T)
    current_states = np.random.normal(0, 1, P)
    current_action = 0

    for t in range(T):
        all_states_mean[t] = np.dot(Gamma, current_states)
        current_states = np.dot(Gamma, current_states) + np.random.normal(0, sigma_residual, P)
        prob_action = 1 / (1 + np.exp(-np.dot(np.hstack([current_action, current_states]), beta_logit)))
        current_action = np.random.binomial(n=1, p=prob_action)
        all_states[t] = current_states
        all_probabilities[t] = prob_action
        all_actions[t] = current_action

    treatment_effect = np.sum(all_states[:, 0:trueP], axis=1) * beta_11 / trueP + baseline_strength
    main_effect = theta1 * np.sum(all_states - all_states_mean, axis=1)

    meanY = main_effect + (all_actions - all_probabilities) * treatment_effect
    errorY = np.dot(C, np.random.normal(size=T))
    #errorY = np.random.normal(size=T)
    obsY = meanY + errorY

    df_individual = pd.DataFrame({
        "id": [id] * T,
        "decision_point": range(1, T + 1),
        **{f"state{i + 1}": all_states[:, i] for i in range(P)},
        "prob": all_probabilities,
        "action": all_actions,
        "outcome": obsY
    })

    return df_individual

def MRT_instance(N=N,
                 T=T,
                 P=P,
                 trueP=trueP):

    individual_data_frames = []

    # Generate individual data and collect them in the list
    for n in range(1, N + 1):
          fake_individual = generate_individual(n, T, P, trueP)  # Call the generate_individual function
          individual_data_frames.append(fake_individual)

    MRT_data = pd.concat(individual_data_frames, ignore_index=True)
    # X1 = np.array(MRT_data.iloc[:201, 2:P+2])
    # Y1 = np.array(MRT_data.iloc[:201, P+4])
    n1 = int(2 * N / 3)
    X1 = MRT_data[MRT_data["id"] > n1].iloc[:, 2:P + 2]
    Y1 = MRT_data[MRT_data["id"] > n1].iloc[:, P + 4]

    alphahat = np.array(sm.OLS(Y1, X1).fit().params)
    Y = np.array(MRT_data[MRT_data["id"] < n1 + 1].iloc[:, P + 4]) - np.dot(np.array(MRT_data[MRT_data["id"] < n1 + 1].iloc[:, 2:P + 2]), alphahat)
    At_Pt = np.array(MRT_data[MRT_data["id"] < n1 + 1].iloc[:, P + 3]) - np.array(MRT_data[MRT_data["id"] < n1 + 1].iloc[:, P + 2])
    X = np.array(MRT_data[MRT_data["id"] < n1 + 1].iloc[:, 2:P + 2].multiply(At_Pt, axis="index"))
    scaling = X.std(0) * np.sqrt(N*T)
    X /= scaling
    
    beta = (beta_11 / trueP) * np.concatenate((np.ones(trueP), np.zeros(P - trueP)))
    A = MRT_data[MRT_data["id"] < n1 + 1].iloc[:, :2]
    A['Y'] = Y.tolist()
    A = A.join(pd.DataFrame(X, columns = ['State'+str(i) for i in range(1,P+1)]))



    # active = np.zeros(P, bool)
    # active[beta != 0] = True

    # scaling = Y.std(0) * np.sqrt(n)
    # Y /= scaling
    return X, Y, beta, A

MRT = MRT_instance(N=750, T=5, P= 100)[3]
# print(X.shape)
# print(Y.shape)
print(MRT)