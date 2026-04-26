import numpy as np
import igraph as ig
from scipy.ndimage import percentile_filter

def preprocess_trace(r, window_size=500, percentile=15):
    """Standardizes r to satisfy rho0=0 and rho1=1."""
    r = np.array(r)
    baseline = percentile_filter(r, percentile=percentile, size=window_size)
    r_proc = (r - baseline) / (baseline + 1e-10)
    return np.maximum(r_proc, 0)

def spp(r_raw, alpha, lmbda=0.1):
    """
    Spike detection using a Dense DAG Shortest Path.
    Input r_raw has length n (r_1 to r_n).
    Returns a binary sequence z of length n-1.
    """
    # 1. Preprocessing
    r = preprocess_trace(r_raw)
    n = len(r) 
    # Potential spike locations: 1, 2, ..., n-1
    # Sink node: n
    
    # 2. Auxiliary Variables (Backward recursion for a_i)
    # a_i defined for i = 1 to n-1
    a = np.zeros(n) # index 1 to n-1 (0 is unused)
    a[n-1] = -r[n-1] # a_{n-1} = -r_n
    for i in range(n - 2, 0, -1):
        a[i] = alpha * a[i+1] - r[i]

    # 3. Construct the Full DAG
    edges = []
    costs = []
    c1 = (1 - alpha**2) / 2.0

    # Case 1: Source edges (i=0 to all j from 1 to n)
    for j in range(1, n + 1):
        edges.append((0, j))
        costs.append(0.0)

    # Case 2: Intermediate edges (1 <= i < j <= n-1)
    for i in range(1, n - 1):
        for j in range(i + 1, n):
            d = j - i
            alpha_d = alpha**d
            alpha_2d = alpha_d**2
            
            num = c1 * (a[i] - alpha_d * a[j])**2
            den = 1 - alpha_2d
            
            costs.append(lmbda - (num / (den + 1e-12)))
            edges.append((i, j))

    # Case 3: Terminal edges (j = n)
    for i in range(1, n):
        d_term = n - i 
        den_term = 1 - alpha**(2 * d_term)
        costs.append(lmbda - (c1 * a[i]**2 / (den_term + 1e-12)))
        edges.append((i, n))

    # 4. Create Graph and Solve
    # Vertices: 0 (Source), 1...(n-1) (Potential Spikes), n (Sink)
    g = ig.Graph(n + 1, edges, directed=True, edge_attrs={'weight': costs})
    path = g.get_shortest_paths(0, to=n, weights="weight", output="vpath")[0]

    # 5. Extract Binary Sequence z*
    # Length is n-1 (corresponding to x_1 to x_{n-1})
    z_star = np.zeros(n - 1, dtype=int)
    for v in path:
        if 1 <= v <= n - 1:
            z_star[v - 1] = 1
    
    return z_star