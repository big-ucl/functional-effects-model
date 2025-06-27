
# import packages
import pandas as pd 
import numpy as np
import lightgbm as lgb
from scipy.special import softmax
from rumboost.datasets import load_preprocess_LPMC

from models_wrapper import RUMBoost, TasteNet

np.random.seed(1)

n_alternatives = 4

def create_discontinuity(x, x_disc, jump):
    return np.where(x < x_disc, x, 0.5*(x-x_disc) + x_disc+jump)

# Define the utility function
def utility_function_LPMC(data, with_noise=False):
    # Extract the parameters
    V = np.zeros((data.shape[0], n_alternatives))

    V[:, 0] = create_functional_intercept(data, ["age", "female"]) + -1 * data['dur_walking']
    V[:, 1] = create_functional_intercept(data, ["age", "car_ownership"]) + -1 * data['dur_cycling']
    V[:, 2] = create_functional_intercept(data, ["age", "driving license"]) + -1 * data['dur_pt']
    V[:, 3] = create_functional_intercept(data, ["female", "car_ownership", "driving_license"]) + -1 * data['dur_driving']

    if with_noise:
        noise = generate_noise(0, 1, (data.shape[0], n_alternatives))
        V += noise

    return V

def generate_noise(mean, sd, n):
    return np.random.gumbel(loc=mean, scale=sd, size=n)

def compute_prob(V):

    return softmax(V, axis=1)

def generate_labels(probs):
    labels = [np.random.choice(range(n_alternatives), p=probs[i]) for i in range(probs.shape[0])]
    return np.array(labels)

def create_functional_intercept(data: pd.DataFrame, features_name: list) -> np.ndarray:
    """
    Create the synthetic functional intercepts

    Parameters
    ----------
    data: pd.DataFrame
        Data used for the synthetic experiment
    features_name: list
        Features used in the functional intercept

    Returns
    -------
    functional_intercept: np.ndarray
        The functional intercepts
    """
    data_arr = data[features_name].values
    functional_intercept = data_arr.prod(axis=1)
    return functional_intercept


if __name__ == "__main__":
    data_train, data_test = load_preprocess_LPMC(path = "../data")
    V_train_noisy = utility_function_LPMC(data_train, with_noise=False)
    V_test_noisy = utility_function_LPMC(data_test, with_noise=True)
    simulated_probs = compute_prob(V_train_noisy)
    simulated_choice = generate_labels(simulated_probs)
    simulated_probs_test = compute_prob(V_test_noisy)
    simulated_choice_test = generate_labels(simulated_probs_test)
    data_train["choice"] = simulated_choice
    data_test["choice"] = simulated_choice_test
    fct_intercept_0 = create_functional_intercept(data_train, ["age", "female"])
    fct_intercept_0_test = create_functional_intercept(data_train, ["age", "female"])
    fct_intercept_1 = create_functional_intercept(data_train, ["age", "car_ownership"])
    fct_intercept_1_test = create_functional_intercept(data_train, ["age", "car_ownership"])
    fct_intercept_2 = create_functional_intercept(data_train, ["age", "driving license"])
    fct_intercept_2_test = create_functional_intercept(data_train, ["age", "driving license"])
    fct_intercept_3 = create_functional_intercept(data_train, ["female", "car_ownership", "driving_license"])
    fct_intercept_3_test = create_functional_intercept(data_train, ["female", "car_ownership", "driving_license"])


    
