# LINUX-SPECIFIC PA

import numpy as np
import multiprocessing as mp

from shell_builders import *


# ============================================================
# build edge indices touching shell nodes
# ============================================================

def build_shell_edge_indices(N, v1, v2, shell_n_nodes, dtype=np.int32):
    """
    For every center node i, precompute edge indices e such that
    edge e = (v1[e], v2[e]) has at least one endpoint in shell_n_nodes[i].

    In other words:

        edge e is included for center i
        if v1[e] in shell_n_nodes[i] OR v2[e] in shell_n_nodes[i]

    Returns
    -------
    shell_edge_indices : list of arrays
        shell_edge_indices[i] gives the edge indices touching shell_n_nodes[i].
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
        d_nodes = shell_n_nodes[i]

        if d_nodes.size == 0:
            shell_edge_indices.append(np.asarray([], dtype=dtype))
            continue

        edge_lists = [
            edges_by_node[int(d)]
            for d in d_nodes
            if edges_by_node[int(d)].size > 0
        ]

        if len(edge_lists) == 0:
            shell_edge_indices.append(np.asarray([], dtype=dtype))
            continue

        # Important:
        # If both endpoints are in the shell, the edge appears twice.
        # We count the edge once.
        edge_idx = np.unique(np.concatenate(edge_lists)).astype(dtype, copy=False)

        shell_edge_indices.append(edge_idx)

    return shell_edge_indices


# ============================================================
# flatten list-of-arrays shell_edge_indices
# ============================================================

def flatten_shell_edge_indices(shell_edge_indices, dtype=np.int32):
    """
    Converts

        shell_edge_indices[i] = array of edge indices

    into

        flat_edge_indices[offsets[i]:offsets[i + 1]]

    This is easier and faster to pass into workers.
    """

    N = len(shell_edge_indices)

    lengths = np.asarray(
        [edge_idx.size for edge_idx in shell_edge_indices],
        dtype=np.int64,
    )

    offsets = np.empty(N + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])

    flat_edge_indices = np.empty(offsets[-1], dtype=dtype)

    for i, edge_idx in enumerate(shell_edge_indices):
        start = offsets[i]
        end = offsets[i + 1]

        if end > start:
            flat_edge_indices[start:end] = edge_idx

    return flat_edge_indices, offsets


# ============================================================
# optional save/load of shell edge structure
# ============================================================

def save_flat_shell_edge_indices(save_path, flat_edge_indices, offsets):
    np.savez_compressed(
        save_path,
        flat_edge_indices=flat_edge_indices,
        offsets=offsets,
    )


def load_flat_shell_edge_indices(load_path):
    with np.load(load_path, allow_pickle=False) as data:
        flat_edge_indices = data["flat_edge_indices"]
        offsets = data["offsets"]

    return flat_edge_indices, offsets


# ============================================================
# compute one snapshot
# ============================================================

def compute_X_and_SI_shell_for_snapshot_flat(
    snapshot_states,
    v1,
    v2,
    flat_edge_indices,
    offsets,
    S_state=0,
    I_state=1,
    dtype_X=np.int8,
    dtype_SI=np.int64,
):
    """
    Computes one realization row:

        X_i = 1 if node i is S, 0 otherwise

        SI_shell_i = number of SI edges with at least one endpoint
                     in shell_n_nodes[i]

    This version uses flattened shell edge indices.
    """

    N = snapshot_states.size

    X_row = (snapshot_states == S_state).astype(dtype_X)

    S = snapshot_states == S_state
    I = snapshot_states == I_state

    # undirected SI edge mask
    SI_edge_mask = (
        (S[v1] & I[v2])
        |
        (I[v1] & S[v2])
    )

    SI_shell_row = np.zeros(N, dtype=dtype_SI)

    nonempty = offsets[1:] > offsets[:-1]

    if np.any(nonempty):
        starts = offsets[:-1][nonempty]

        values = SI_edge_mask[flat_edge_indices].astype(dtype_SI, copy=False)

        SI_shell_row[nonempty] = np.add.reduceat(
            values,
            starts,
        )

    return X_row, SI_shell_row


# ============================================================
# load snapshot from full process
# ============================================================

def load_snapshot_at_or_before_time(full_process_path, matched_curve_time):
    """
    Loads one full_process file and returns the last raw snapshot with

        raw_time <= matched_curve_time

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


# ============================================================
# serial matrix builder, kept for checking/debugging
# ============================================================

def build_X_and_SI_shell_matrices_serial(
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
    Original-style serial version.

    Builds the full matrices:

        X_matrix.shape        = (N_loaded, N)
        SI_shell_matrix.shape = (N_loaded, N)

    Use this only for small N.
    """

    edge_dtype = np.int32
    if len(v1) > np.iinfo(np.int32).max:
        edge_dtype = np.int64

    shell_edge_indices = build_shell_edge_indices(
        N=N,
        v1=v1,
        v2=v2,
        shell_n_nodes=shell_n_nodes,
        dtype=edge_dtype,
    )

    flat_edge_indices, offsets = flatten_shell_edge_indices(
        shell_edge_indices,
        dtype=edge_dtype,
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

        X_row, SI_shell_row = compute_X_and_SI_shell_for_snapshot_flat(
            snapshot_states=snapshot_states,
            v1=v1,
            v2=v2,
            flat_edge_indices=flat_edge_indices,
            offsets=offsets,
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


# ============================================================
# summarize full matrices
# ============================================================

def summarize_covariance_from_X_and_SI_shell(X_matrix, SI_shell_matrix):
    """
    Computes

        sum_i <X_i SI_shell_i>
        sum_i <X_i> <SI_shell_i>
        covariance = sum_i Cov(X_i, SI_shell_i)
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


# ============================================================
# multiprocessing worker setup
# ============================================================

_WORKER_DATA = {}


def init_worker(
    matched_curve_time,
    v1,
    v2,
    flat_edge_indices,
    offsets,
    S_state,
    I_state,
):
    """
    Initializer for multiprocessing workers.

    On Linux with fork, these arrays are shared copy-on-write,
    so they are not fully copied for every worker unless modified.
    """

    global _WORKER_DATA

    _WORKER_DATA = {
        "matched_curve_time": matched_curve_time,
        "v1": v1,
        "v2": v2,
        "flat_edge_indices": flat_edge_indices,
        "offsets": offsets,
        "S_state": S_state,
        "I_state": I_state,
    }


def process_path_chunk(full_process_path_chunk):
    """
    Worker function.

    Processes several full_process files and returns accumulated sums:

        sum_X
        sum_SI_shell
        sum_X_times_SI_shell

    This avoids returning full X/SI matrices.
    """

    global _WORKER_DATA

    matched_curve_time = _WORKER_DATA["matched_curve_time"]
    v1 = _WORKER_DATA["v1"]
    v2 = _WORKER_DATA["v2"]
    flat_edge_indices = _WORKER_DATA["flat_edge_indices"]
    offsets = _WORKER_DATA["offsets"]
    S_state = _WORKER_DATA["S_state"]
    I_state = _WORKER_DATA["I_state"]

    N = offsets.size - 1

    sum_X = np.zeros(N, dtype=np.float64)
    sum_SI_shell = np.zeros(N, dtype=np.float64)
    sum_X_times_SI_shell = np.zeros(N, dtype=np.float64)

    snapshot_times = []
    loaded_paths = []

    N_loaded = 0

    for full_process_path in full_process_path_chunk:
        loaded, snapshot_time, snapshot_states = load_snapshot_at_or_before_time(
            full_process_path=full_process_path,
            matched_curve_time=matched_curve_time,
        )

        if not loaded:
            continue

        X_row, SI_shell_row = compute_X_and_SI_shell_for_snapshot_flat(
            snapshot_states=snapshot_states,
            v1=v1,
            v2=v2,
            flat_edge_indices=flat_edge_indices,
            offsets=offsets,
            S_state=S_state,
            I_state=I_state,
            dtype_X=np.int8,
            dtype_SI=np.int64,
        )

        sum_X += X_row
        sum_SI_shell += SI_shell_row
        sum_X_times_SI_shell += X_row * SI_shell_row

        snapshot_times.append(snapshot_time)
        loaded_paths.append(full_process_path)

        N_loaded += 1

    return (
        N_loaded,
        sum_X,
        sum_SI_shell,
        sum_X_times_SI_shell,
        snapshot_times,
        loaded_paths,
    )


def chunk_list(x, chunk_size):
    return [
        x[i:i + chunk_size]
        for i in range(0, len(x), chunk_size)
    ]


# ============================================================
# main parallel covariance summary
# ============================================================

def summarize_covariance_parallel_streaming(
    full_process_paths,
    matched_curve_time,
    N,
    v1,
    v2,
    shell_n_nodes=None,
    flat_edge_indices=None,
    offsets=None,
    S_state=0,
    I_state=1,
    n_workers=8,
    chunk_size=20,
    multiprocessing_context="fork",
):
    """
    Parallel streaming version.

    This does NOT build:

        X_matrix
        SI_shell_matrix

    Instead it accumulates:

        sum_r X_i^(r)
        sum_r SI_shell_i^(r)
        sum_r X_i^(r) SI_shell_i^(r)

    Then it computes the same covariance quantities.

    You must provide either:

        shell_n_nodes

    or the precomputed pair:

        flat_edge_indices, offsets
    """

    if flat_edge_indices is None or offsets is None:
        if shell_n_nodes is None:
            raise ValueError(
                "Provide either shell_n_nodes or precomputed flat_edge_indices and offsets."
            )

        edge_dtype = np.int32
        if len(v1) > np.iinfo(np.int32).max:
            edge_dtype = np.int64

        print("Building shell_edge_indices...")

        shell_edge_indices = build_shell_edge_indices(
            N=N,
            v1=v1,
            v2=v2,
            shell_n_nodes=shell_n_nodes,
            dtype=edge_dtype,
        )

        print("Flattening shell_edge_indices...")

        flat_edge_indices, offsets = flatten_shell_edge_indices(
            shell_edge_indices,
            dtype=edge_dtype,
        )

    path_chunks = chunk_list(full_process_paths, chunk_size)

    total_loaded = 0

    total_sum_X = np.zeros(N, dtype=np.float64)
    total_sum_SI_shell = np.zeros(N, dtype=np.float64)
    total_sum_X_times_SI_shell = np.zeros(N, dtype=np.float64)

    all_snapshot_times = []
    all_loaded_paths = []

    ctx = mp.get_context(multiprocessing_context)

    with ctx.Pool(
        processes=n_workers,
        initializer=init_worker,
        initargs=(
            matched_curve_time,
            v1,
            v2,
            flat_edge_indices,
            offsets,
            S_state,
            I_state,
        ),
    ) as pool:

        for result in pool.imap_unordered(process_path_chunk, path_chunks):
            (
                N_loaded,
                sum_X,
                sum_SI_shell,
                sum_X_times_SI_shell,
                snapshot_times,
                loaded_paths,
            ) = result

            if N_loaded == 0:
                continue

            total_loaded += N_loaded
            total_sum_X += sum_X
            total_sum_SI_shell += sum_SI_shell
            total_sum_X_times_SI_shell += sum_X_times_SI_shell

            all_snapshot_times.extend(snapshot_times)
            all_loaded_paths.extend(loaded_paths)

            # print(
            #     "Loaded snapshots so far:",
            #     total_loaded,
            #     "/",
            #     len(full_process_paths),
            # )

    if total_loaded == 0:
        raise RuntimeError(
            "No snapshots were loaded. matched_curve_time is before the first raw time in all processes."
        )

    mean_X = total_sum_X / total_loaded
    mean_SI_shell = total_sum_SI_shell / total_loaded
    mean_X_times_SI_shell = total_sum_X_times_SI_shell / total_loaded

    per_node_covariance = mean_X_times_SI_shell - mean_X * mean_SI_shell

    mean_S_SI_nth = np.sum(mean_X_times_SI_shell)
    factorized = np.sum(mean_X * mean_SI_shell)
    covariance = np.sum(per_node_covariance)

    snapshot_times = np.asarray(all_snapshot_times, dtype=np.float64)

    return {
        "N_loaded": total_loaded,
        "mean_X": mean_X,
        "mean_SI_shell": mean_SI_shell,
        "mean_X_times_SI_shell": mean_X_times_SI_shell,
        "per_node_covariance": per_node_covariance,
        "mean_S_SI_nth": mean_S_SI_nth,
        "factorized": factorized,
        "covariance": covariance,
        "relative_covariance": covariance / factorized if factorized != 0 else np.nan,
        "snapshot_times": snapshot_times,
        "loaded_paths": all_loaded_paths,
        "flat_edge_indices": flat_edge_indices,
        "offsets": offsets,
    }


# ============================================================
# small helper for printing summary
# ============================================================

def print_covariance_summary(summary):
    print()
    print("Covariance summary")
    print("------------------")
    print("N_loaded              =", summary["N_loaded"])
    print("mean_S_SI_nth         =", summary["mean_S_SI_nth"])
    print("factorized            =", summary["factorized"])
    print("covariance            =", summary["covariance"])
    print("relative_covariance   =", summary["relative_covariance"])
    print()


# ============================================================
# example usage
# ============================================================

if __name__ == "__main__":

    # This block is only a template.
    # Usually you will already have these variables from your analysis script:
    #
    #   full_process_paths
    #   matched_curve_time
    #   N
    #   v1
    #   v2
    #   shell_n_nodes
    #
    # Then call the function below.

    raise SystemExit(
        "Import this file and call summarize_covariance_parallel_streaming(...) "
        "from your analysis script."
    )