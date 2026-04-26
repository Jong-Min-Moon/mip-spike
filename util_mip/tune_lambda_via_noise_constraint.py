import numpy as np
from .estimate_alpha_ar1 import estimate_alpha_ar1
from .estimate_noise_variance_psd import estimate_noise_variance_psd
from .reconstruct_optimal_s import reconstruct_optimal_s

def tune_lambda_via_noise_constraint(y, estimator, tol=1e-3, max_iter=50):
    """
    Finds the optimal lambda using a bisection search based on a noise constraint.
    """
    alpha = estimate_alpha_ar1(y)
    estimated_noise_var = estimate_noise_variance_psd(y)
    r = y['dff']
    n = len(r)

    # The physical noise floor we want our model's residual to hit
    target_rss = n * estimated_noise_var

    # Initialize the Bisection Search bounds
    lambda_low = 0.000001
    lambda_high = 3000.0  # Make sure this is high enough to suppress all spikes

    print(f"Target RSS: {target_rss:.4f}")

    optimal_lambda = None
    best_s, best_z = None, None

    for i in range(max_iter):
        # 1. Guess the midpoint
        lam_mid = (lambda_low + lambda_high) / 2.0

        # 2. Solve the DAG with this lambda
        z_pred = estimator(r, alpha, lam_mid)

        # 3. Reconstruct the optimal continuous state s directly from z
        s_pred = reconstruct_optimal_s(r, z_pred, alpha)

        # 4. Calculate the current RSS
        current_rss = np.sum((s_pred - r)**2)

        error = current_rss - target_rss
        print(f"Iter {i:2d}: lambda={lam_mid:.10f} | RSS={current_rss:.6f} | Error={error:.6f}")

        # 5. Check for convergence
        if abs(error) / target_rss < tol:
            optimal_lambda = lam_mid
            best_s, best_z = s_pred, z_pred
            break

        # 6. Update Bisection Bounds
        if current_rss < target_rss:
            # RSS is too low. We are overfitting the noise.
            # We need FEWER spikes, so we must INCREASE lambda.
            lambda_low = lam_mid
        else:
            # RSS is too high. We are underfitting.
            # We need MORE spikes, so we must DECREASE lambda.
            lambda_high = lam_mid

    if optimal_lambda is None:
        print("Warning: Bisection hit max iterations without strict convergence.")
        optimal_lambda = lam_mid
        best_s, best_z = s_pred, z_pred

    return optimal_lambda, best_s, best_z
