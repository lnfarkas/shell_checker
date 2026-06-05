import numpy as np


def build_neighbors_from_edges(N, v1, v2, dtype=np.int64):
    neighbors = [[] for _ in range(N)]

    for u, v in zip(v1, v2):
        neighbors[u].append(v)
        neighbors[v].append(u)

    return [np.asarray(x, dtype=dtype) for x in neighbors]


def build_shell_n_nodes_set(N, neighbors, shell_n, dtype=np.int32):
    """
    shell_n_nodes[a] = array of nodes at graph distance exactly shell_n from node a

    Set/frontier version.
    """

    shell_n_nodes = []

    for a in range(N):
        visited = set([a])
        current_shell = set([a])

        for _ in range(shell_n):
            next_shell = set()

            for u in current_shell:
                next_shell.update(neighbors[u])

            next_shell.difference_update(visited)

            visited.update(next_shell)
            current_shell = next_shell

        shell_n_nodes.append(np.fromiter(current_shell, dtype=dtype))

    return shell_n_nodes


def build_shell_n_nodes_dense_numpy(N, v1, v2, shell_n, dtype=np.int32):
    """
    Dense NumPy version.

    B_k = A^k > 0

    shell_n = B_n AND NOT(I OR B_1 OR ... OR B_{n-1})

    Warning:
        Builds dense N x N matrices.
        Good for testing N ~ 1000.
        Bad for very large N.
    """

    A_bool = np.zeros((N, N), dtype=bool)
    A_bool[v1, v2] = True
    A_bool[v2, v1] = True

    A_int = A_bool.astype(np.int32)

    I = np.eye(N, dtype=bool)

    if shell_n == 0:
        shell_matrix = I
    else:
        B = A_bool.copy()
        reached_closer = I.copy()

        for k in range(1, shell_n + 1):
            if k > 1:
                B = (B.astype(np.int32) @ A_int) > 0

            if k < shell_n:
                reached_closer |= B

        shell_matrix = B & ~reached_closer

    shell_n_nodes = [
        np.flatnonzero(shell_matrix[a]).astype(dtype)
        for a in range(N)
    ]

    return shell_n_nodes


def shell_lists_equal(shell_a, shell_b, verbose=True):
    """
    Compare two shell_n_nodes lists, ignoring order inside each shell.
    """

    if len(shell_a) != len(shell_b):
        if verbose:
            print(f"Different number of nodes: {len(shell_a)} vs {len(shell_b)}")
        return False

    for i, (x, y) in enumerate(zip(shell_a, shell_b)):
        x_sorted = np.sort(x)
        y_sorted = np.sort(y)

        if not np.array_equal(x_sorted, y_sorted):
            if verbose:
                print(f"\nMismatch at node {i}")
                print("first :", x_sorted)
                print("second:", y_sorted)
            return False

    return True


def shell_size_summary(shells):
    sizes = np.asarray([x.size for x in shells], dtype=np.int64)

    print(f"mean shell size = {np.mean(sizes):.6f}")
    print(f"min shell size  = {np.min(sizes)}")
    print(f"max shell size  = {np.max(sizes)}")