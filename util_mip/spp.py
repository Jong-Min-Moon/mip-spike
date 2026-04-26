import numpy as np
import graph_tool.all as gt

def spp(r, alpha, lmbda=0.1):
    n = len(r)
    
    # 1. Auxiliary Variables
    a = np.zeros(n) 
    a[n-1] = -r[n-1] 
    for i in range(n - 2, 0, -1):
        a[i] = alpha * a[i+1] - r[i]

    # 2. Initialize Graph
    g = gt.Graph(directed=True)
    g.add_vertex(n + 1) # This creates vertices with indices 0...n
    
    weights = g.new_edge_property("double")
    c1 = (1 - alpha**2) / 2.0
    
    edge_list = []
    weight_list = []

    # Case 1: Source edges
    for j in range(1, n + 1):
        edge_list.append((0, j))
        weight_list.append(0.0)

    # Case 2: Intermediate edges
    for i in range(1, n - 1):
        for j in range(i + 1, n):
            d = j - i
            alpha_d = alpha**d
            alpha_2d = alpha_d**2
            num = c1 * (a[i] - alpha_d * a[j])**2
            den = 1 - alpha_2d
            edge_list.append((i, j))
            weight_list.append(lmbda - (num / (den + 1e-12)))

    # Case 3: Terminal edges
    for i in range(1, n):
        d_term = n - i
        den_term = 1 - alpha**(2 * d_term)
        edge_list.append((i, n))
        weight_list.append(lmbda - (c1 * a[i]**2 / (den_term + 1e-12)))

    # 3. Add edges and assign weights
    g.add_edge_list(edge_list)
    weights.a = np.array(weight_list)

    # 4. Solve Shortest Path
    # Using g.vertex(index) is the safest way to pass source/target
    vertex_list, _ = gt.shortest_path(g, source=g.vertex(0), target=g.vertex(n), 
                                      weights=weights, dag=True)

    # 5. Extract Binary Sequence z*
    z_star = np.zeros(n - 1, dtype=int)
    for node in vertex_list:
        v_idx = int(node)
        if 1 <= v_idx <= n - 1:
            z_star[v_idx - 1] = 1

    return z_star