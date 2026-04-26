import numpy as np

def reconstruct_optimal_s(r, z, alpha):
    """
    Computes the globally optimal calcium trace s using independent OLS blocks.
    Strictly guarantees that len(s) == len(r).
    """
    n = len(r)
    s = np.zeros(n) # Force s to be exactly the same length as r

    # Find the indices where blocks start (where z == 1)
    spike_indices = np.where(z == 1)[0]
    block_starts = list(spike_indices)

    # Ensure the first block starts at index 0 even if there's no spike there
    if len(block_starts) == 0 or block_starts[0] != 0:
        block_starts.insert(0, 0)

    # Cap the final block at the end of the array
    block_starts.append(n)

    for b in range(len(block_starts) - 1):
        start_idx = block_starts[b]
        end_idx = block_starts[b+1]
        L = end_idx - start_idx

        r_block = r[start_idx:end_idx]

        powers = np.arange(L)
        alpha_t = alpha ** powers

        numerator = np.sum(alpha_t * r_block)
        denominator = np.sum(alpha_t ** 2)

        # OLS estimate for the block amplitude, strictly non-negative
        optimal_amplitude = max(0.0, numerator / denominator)

        # Populate the state trajectory
        s[start_idx:end_idx] = optimal_amplitude * alpha_t

    return s
