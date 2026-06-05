import numpy as np
from shell_builders import *

def build_shell_edge_indices(N, v1, v2, shell_n_nodes, dtype=np.int32):
    """
    For every center node i, precompute the edge indices e such that
    edge e = (v1[e], v2[e]) has at least one endpoint in shell_n_nodes[i].

    This corresponds to the condition:

        (j in nth_i and ell not in nth_i)
        OR
        (j not in nth_i and ell in nth_i)
        OR
        (j in nth_i and ell in nth_i)

    In other words:
        at least one endpoint of the edge is in the nth shell of i.

    Returns
    -------
    shell_edge_indices : list of arrays
        shell_edge_indices[i] gives the edge indices touching the nth shell of i.
    """

    E = len(v1)

    edges_by_node = [[] for _ in range(N)]

    for e in range(E):
        u = int(v1[e])
        v = int(v2[e])

        edges_by_node[u].append(e)
        edges_by_node[v].append(e)

    edges_by_node = [
        np.asarray(x, dtype=dtype)
        for x in edges_by_node
    ]

    shell_edge_indices = []

    for i in range(N):
        d_nodes = shell_n_nodes[i] # name comes from a-b-c-d example, where we are on a and the node d is in shell 3

        if d_nodes.size == 0:
            shell_edge_indices.append(np.asarray([], dtype=dtype))
            continue

        # collect all edge indices incident to at least one shell node
        edge_lists = [
            edges_by_node[d]
            for d in d_nodes
            if edges_by_node[d].size > 0
        ]

        if len(edge_lists) == 0:
            shell_edge_indices.append(np.asarray([], dtype=dtype))
            continue

        # unique is important:
        # if both endpoints are in the shell, the edge appears twice;
        # the formula counts it once.
        edge_idx = np.unique(np.concatenate(edge_lists)).astype(dtype)

        shell_edge_indices.append(edge_idx)

    return shell_edge_indices


def compute_X_and_SI_shell_for_snapshot(
    snapshot_states,
    v1,
    v2,
    shell_edge_indices,
    S_state=0,
    I_state=1,
    dtype_X=np.int8,
    dtype_SI=np.int64,
):
    """
    Computes one row of:

        X_matrix[r, i]        = X_i^(r)
        SI_shell_matrix[r, i] = ([SI]_{nth_i})^(r)

    for one snapshot / realization.

    X_i^(r) = 1 if node i is S, 0 otherwise.

    ([SI]_{nth_i})^(r) is the number of SI edges where at least one endpoint
    of the SI edge lies in shell_n_nodes[i].
    """

    N = snapshot_states.size

    X_row = (snapshot_states == S_state).astype(dtype_X)

    S = snapshot_states == S_state
    I = snapshot_states == I_state

    # undirected SI edge mask:
    # edge e is SI if one endpoint is S and the other is I
    SI_edge_mask = (
        (S[v1] & I[v2])
        |
        (I[v1] & S[v2])
    )

    SI_shell_row = np.zeros(N, dtype=dtype_SI)

    for i in range(N):
        edge_idx = shell_edge_indices[i]

        if edge_idx.size == 0:
            continue

        SI_shell_row[i] = np.sum(SI_edge_mask[edge_idx])

    return X_row, SI_shell_row


def load_snapshot_at_or_before_time(full_process_path, matched_curve_time):
    """
    Loads one full_process file and returns the last raw snapshot with

        raw_time <= matched_curve_time 
        (we give a time to the code - it looks at the mean_S curves projected to common time axis,
        finds nearest time on the axis, then takes raw runs where actual and not common projected times are noted
        and takes from snapshots the state after the last jump before the projected matched time)

    Returns
    -------
    loaded : bool
    snapshot_time : float or None
    snapshot_states : ndarray or None
    """

    with np.load(full_process_path, allow_pickle=False) as process_data:
        raw_times = process_data["raw_times"]
        raw_vertex_states = process_data["raw_vertex_states"]

        row_index = int(
            np.searchsorted(raw_times, matched_curve_time, side="right") - 1
        )

        if row_index < 0:
            return False, None, None

        snapshot_time = float(raw_times[row_index])
        snapshot_states = raw_vertex_states[row_index].copy()

    return True, snapshot_time, snapshot_states


def build_X_and_SI_shell_matrices(
    full_process_paths,
    matched_curve_time,
    N,
    v1,
    v2,
    shell_n_nodes,
    S_state=0,
    I_state=1,
    dtype_X=np.int8,
    dtype_SI=np.int64,
):
    """
    Builds the two N_r x N matrices:

        X_matrix[r, i]        = X_i^(r)
        SI_shell_matrix[r, i] = ([SI]_{nth_i})^(r)

    Only processes with at least one raw_time <= matched_curve_time are loaded.

    Returns
    -------
    X_matrix : ndarray, shape (N_loaded, N)
    SI_shell_matrix : ndarray, shape (N_loaded, N)
    snapshot_times : ndarray, shape (N_loaded,)
    loaded_paths : list of Path
    """

    shell_edge_indices = build_shell_edge_indices(
        N=N,
        v1=v1,
        v2=v2,
        shell_n_nodes=shell_n_nodes,
        dtype=np.int32,
    )

    X_rows = []
    SI_shell_rows = []
    snapshot_times = []
    loaded_paths = []

    for full_process_path in full_process_paths:
        loaded, snapshot_time, snapshot_states = load_snapshot_at_or_before_time(
            full_process_path=full_process_path,
            matched_curve_time=matched_curve_time,
        )

        if not loaded:
            continue

        X_row, SI_shell_row = compute_X_and_SI_shell_for_snapshot(
            snapshot_states=snapshot_states,
            v1=v1,
            v2=v2,
            shell_edge_indices=shell_edge_indices,
            S_state=S_state,
            I_state=I_state,
            dtype_X=dtype_X,
            dtype_SI=dtype_SI,
        )

        X_rows.append(X_row)
        SI_shell_rows.append(SI_shell_row)
        snapshot_times.append(snapshot_time)
        loaded_paths.append(full_process_path)

    if len(X_rows) == 0:
        raise RuntimeError(
            "No snapshots were loaded. matched_curve_time is before the first raw time in all processes."
        )

    X_matrix = np.vstack(X_rows).astype(dtype_X, copy=False)
    SI_shell_matrix = np.vstack(SI_shell_rows).astype(dtype_SI, copy=False)
    snapshot_times = np.asarray(snapshot_times, dtype=np.float64)

    return X_matrix, SI_shell_matrix, snapshot_times, loaded_paths


def summarize_covariance_from_X_and_SI_shell(X_matrix, SI_shell_matrix):
    """
    Given

        X_matrix[r, i]        = X_i^(r)
        SI_shell_matrix[r, i] = ([SI]_{nth_i})^(r)

    computes:

        mean_S_SI_nth =
            (1/N_r) sum_r sum_i X_i^(r) ([SI]_{nth_i})^(r)

        factorized =
            sum_i <X_i> <[SI]_{nth_i}>

        covariance =
            mean_S_SI_nth - factorized
    """

    mean_X = np.mean(X_matrix, axis=0)
    mean_SI_shell = np.mean(SI_shell_matrix, axis=0)

    mean_X_times_SI_shell = np.mean(
        X_matrix.astype(np.float64) * SI_shell_matrix.astype(np.float64),
        axis=0,
    )

    per_node_covariance = mean_X_times_SI_shell - mean_X * mean_SI_shell

    mean_S_SI_nth = np.sum(mean_X_times_SI_shell)
    factorized = np.sum(mean_X * mean_SI_shell)
    covariance = np.sum(per_node_covariance)

    return {
        "mean_X": mean_X,
        "mean_SI_shell": mean_SI_shell,
        "mean_X_times_SI_shell": mean_X_times_SI_shell,
        "per_node_covariance": per_node_covariance,
        "mean_S_SI_nth": mean_S_SI_nth,
        "factorized": factorized,
        "covariance": covariance,
        "relative_covariance": covariance / factorized if factorized != 0 else np.nan,
    }