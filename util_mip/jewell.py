import numpy as np
from FastLZeroSpikeInference import fast

def jewell(y, gam, lam):
    # Run the estimation
    fit = fast.estimate_spikes(y, gam, lam, False)

    # Create a dense binary vector of zeros with the same length as the input trace
    z = np.zeros(len(y), dtype=int)

    # Extract the spike indices
    spike_indices = fit['spikes']

    # If any spikes were found, set their corresponding indices in z to 1
    if len(spike_indices) > 0:
        # Cast to int just in case the library returns floats
        valid_indices = np.array(spike_indices).astype(int)
        z[valid_indices] = 1

    return z