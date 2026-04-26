import numpy as np

def estimate_alpha_ar1(y):
    """
    Estimates the AR(1) coefficient alpha from the data y.
    """
    r = y['dff']
    N = len(r)
    if N <= 2:
        raise ValueError("The trace must contain more than 2 observations to compute lag-2 autocovariance.")

    # Calculate the empirical mean (\bar{r}) and center the data
    r_bar = np.mean(r)
    r_centered = r - r_bar

    # Compute the empirical autocovariance at lag \tau = 1 (\hat{C}_r(1))
    C_hat_1 = np.sum(r_centered[:-1] * r_centered[1:]) / (N - 1)

    # Compute the empirical autocovariance at lag \tau = 2 (\hat{C}_r(2))
    C_hat_2 = np.sum(r_centered[:-2] * r_centered[2:]) / (N - 2)

    # Prevent division by zero if the trace is perfectly flat/uncorrelated
    if C_hat_1 == 0:
        raise ZeroDivisionError(r"Autocovariance at lag 1 is exactly zero; \hat{\alpha} is undefined.")

    # Calculate and return \hat{\alpha}
    alpha_hat = C_hat_2 / C_hat_1

    return alpha_hat
