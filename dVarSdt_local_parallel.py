import numpy as np
from pathlib import Path
import time
import matplotlib.pyplot as plt
import multiprocessing as mp

from concurrent.futures import ProcessPoolExecutor, as_completed

from loaders import *
from shell_covariance_parallel import *

# ============================================================
# choose graph
# ============================================================

simID = "SIsimUNDIRECTED20260604163133"  # N = 1000

# simID = "SIsimUNDIRECTED20260604143341"  # N = 100
# simID = "SIsimUNDIRECTED20260604123046"  # N = 10

base_dir = Path(f"/home/lnf/Desktop/00_sim_SI/{simID}")
graphs_dir = base_dir / "Graphs"

INSTANCE_NUMBER = 0
INSTANCE_TAG = f"instanceNo{INSTANCE_NUMBER:04d}"

beta = 1.0

full_process_dir = base_dir / "FullProcess"
curves_dir = base_dir / "Curves"

N_processes = 1000
approx_time = 0.3

S_state = 0
I_state = 1

# For .npz files, 30 can be too many because the disk becomes the bottleneck.
# Try 8, 12, 16, 19 and compare.
n_workers = 19
n_chunks_multiplier = 4

SAVE_FIGURES = True

# ============================================================
# globals used by worker processes
# ============================================================

N = None
T = None
E = None

v1 = None
v2 = None
time_grid = None

common_edge_ids = None
common_nodes_flat = None

mean_X = None
local_mean_S_around_edge_all = None


# ============================================================
# helper functions
# ============================================================

def make_chunks(items, n_chunks):
    items = list(items)
    n = len(items)

    if n == 0:
        return []

    n_chunks = max(1, min(n_chunks, n))
    chunk_size = int(np.ceil(n / n_chunks))

    return [
        items[i:i + chunk_size]
        for i in range(0, n, chunk_size)
    ]


def load_full_process_file(path):
    """
    Loads one full-process .npz file.

    Your files use:
        raw_times
        raw_vertex_states
    """

    with np.load(path, allow_pickle=True) as d:
        times = np.asarray(d["raw_times"], dtype=np.float64)
        states_all = np.asarray(d["raw_vertex_states"])

    # Make sure shape is (n_times, N)
    if states_all.ndim != 2:
        raise ValueError(
            f"raw_vertex_states should be 2D, but got shape {states_all.shape} in {path}"
        )

    if states_all.shape[0] != len(times) and states_all.shape[1] == len(times):
        states_all = states_all.T

    if states_all.shape[0] != len(times):
        raise ValueError(
            f"State/time shape mismatch in {path}: "
            f"states shape = {states_all.shape}, len(times) = {len(times)}"
        )

    return times, states_all


def build_common_neighbor_flat(v1_arr, v2_arr, n_vertices):
    """
    Builds a vectorized representation of common neighbours for each edge.

    This replaces the slow per-edge Python loop inside closed_edge_local_sum.
    """

    neighbors = [set() for _ in range(n_vertices)]

    for a, b in zip(v1_arr, v2_arr):
        neighbors[a].add(b)
        neighbors[b].add(a)

    edge_id_parts = []
    node_parts = []

    for e, (a, b) in enumerate(zip(v1_arr, v2_arr)):
        common = neighbors[a] & neighbors[b]

        if len(common) == 0:
            continue

        common_nodes = np.fromiter(common, dtype=np.int64, count=len(common))
        common_edges = np.full(len(common_nodes), e, dtype=np.int64)

        node_parts.append(common_nodes)
        edge_id_parts.append(common_edges)

    if len(node_parts) == 0:
        common_nodes_flat_out = np.empty(0, dtype=np.int64)
        common_edge_ids_out = np.empty(0, dtype=np.int64)
    else:
        common_nodes_flat_out = np.concatenate(node_parts)
        common_edge_ids_out = np.concatenate(edge_id_parts)

    return common_edge_ids_out, common_nodes_flat_out


def closed_edge_local_sum(x):
    """
    For every edge e=(a,b), return

        sum_{i in {a,b} union N(a) union N(b)} x_i

    Equivalently, for edge (a,b):

        sum_N(a) + sum_N(b) - sum_common_neighbors(a,b)

    Since a and b are adjacent, this includes a and b automatically.
    """

    global N, E, v1, v2, common_edge_ids, common_nodes_flat

    x = np.asarray(x, dtype=np.float64)

    nbr_sum = np.zeros(N, dtype=np.float64)

    np.add.at(nbr_sum, v1, x[v2])
    np.add.at(nbr_sum, v2, x[v1])

    if common_nodes_flat.size > 0:
        common_sum = np.bincount(
            common_edge_ids,
            weights=x[common_nodes_flat],
            minlength=E,
        )
    else:
        common_sum = np.zeros(E, dtype=np.float64)

    return nbr_sum[v1] + nbr_sum[v2] - common_sum


def compute_snapshot_and_local_chunk(paths_chunk):
    """
    Combined pass.

    This replaces:
        - old first parallel snapshot-average pass
        - old serial PASS 1

    So full-process files are read once here instead of twice.
    """

    global N, T, v1, v2, time_grid

    local_sum_S = np.zeros(T, dtype=np.float64)
    local_sum_S2 = np.zeros(T, dtype=np.float64)
    local_sum_SI = np.zeros(T, dtype=np.float64)
    local_sum_S_SI = np.zeros(T, dtype=np.float64)

    local_sum_X = np.zeros((T, N), dtype=np.float64)
    local_sum_S_SI_local = np.zeros(T, dtype=np.float64)

    local_n = np.zeros(T, dtype=np.int64)

    for path in paths_chunk:
        times, states_all = load_full_process_file(path)

        snapshot_indices = np.searchsorted(times, time_grid, side="right") - 1

        for tidx, snap_idx in enumerate(snapshot_indices):
            if snap_idx < 0:
                continue

            states = states_all[snap_idx]

            S = states == S_state
            I = states == I_state

            edge_v1_S_v2_I = S[v1] & I[v2]
            edge_v2_S_v1_I = S[v2] & I[v1]
            edge_is_SI = edge_v1_S_v2_I | edge_v2_S_v1_I

            S_count = np.sum(S)
            SI_count = np.sum(edge_is_SI)

            local_S_around_edge = closed_edge_local_sum(S)
            S_SI_local = np.sum(edge_is_SI * local_S_around_edge)

            local_sum_S[tidx] += S_count
            local_sum_S2[tidx] += S_count**2
            local_sum_SI[tidx] += SI_count
            local_sum_S_SI[tidx] += S_count * SI_count

            local_sum_X[tidx] += S
            local_sum_S_SI_local[tidx] += S_SI_local

            local_n[tidx] += 1

    return (
        local_sum_S,
        local_sum_S2,
        local_sum_SI,
        local_sum_S_SI,
        local_sum_X,
        local_sum_S_SI_local,
        local_n,
    )


def compute_factorized_chunk(paths_chunk):
    """
    Parallel version of old PASS 2.

    Uses precomputed local_mean_S_around_edge_all[tidx], because this depends only
    on mean_X[tidx], not on the realization.
    """

    global T, v1, v2, time_grid, mean_X, local_mean_S_around_edge_all

    local_sum_factorized_local = np.zeros(T, dtype=np.float64)
    local_sum_factorized_S_on_I = np.zeros(T, dtype=np.float64)
    local_n = np.zeros(T, dtype=np.int64)

    for path in paths_chunk:
        times, states_all = load_full_process_file(path)

        snapshot_indices = np.searchsorted(times, time_grid, side="right") - 1

        for tidx, snap_idx in enumerate(snapshot_indices):
            if snap_idx < 0:
                continue

            states = states_all[snap_idx]

            S = states == S_state
            I = states == I_state

            edge_v1_S_v2_I = S[v1] & I[v2]
            edge_v2_S_v1_I = S[v2] & I[v1]
            edge_is_SI = edge_v1_S_v2_I | edge_v2_S_v1_I

            local_sum_factorized_local[tidx] += np.sum(
                edge_is_SI * local_mean_S_around_edge_all[tidx]
            )

            local_sum_factorized_S_on_I[tidx] += (
                np.sum(edge_v1_S_v2_I * mean_X[tidx, v2])
                + np.sum(edge_v2_S_v1_I * mean_X[tidx, v1])
            )

            local_n[tidx] += 1

    return local_sum_factorized_local, local_sum_factorized_S_on_I, local_n


def save_and_show(fig, fig_path=None):
    if SAVE_FIGURES and fig_path is not None:
        fig.savefig(fig_path, dpi=250, bbox_inches="tight")

    plt.show()
    plt.close(fig)


# ============================================================
# main
# ============================================================

def main():
    global N, T, E
    global v1, v2, time_grid
    global common_edge_ids, common_nodes_flat
    global mean_X, local_mean_S_around_edge_all

    try:
        mp.set_start_method("fork", force=True)
    except Exception:
        pass

    t_total_start = time.time()

    # ============================================================
    # load graph
    # ============================================================

    graph_path, N_loaded, v1_loaded, v2_loaded = load_graph_file(
        graphs_dir=graphs_dir,
        instance_tag=INSTANCE_TAG,
    )

    N = int(N_loaded)
    v1 = np.asarray(v1_loaded, dtype=np.int64)
    v2 = np.asarray(v2_loaded, dtype=np.int64)
    E = len(v1)

    common_edge_ids, common_nodes_flat = build_common_neighbor_flat(v1, v2, N)

    # ============================================================
    # find full process paths and load curve file
    # ============================================================

    full_process_paths = find_full_process_paths(
        full_process_dir=full_process_dir,
        instance_tag=INSTANCE_TAG,
        n_processes=N_processes,
    )

    full_process_paths = [str(p) for p in full_process_paths]

    curve_path, curve_idx, matched_curve_time, curve_data = load_curve_and_match_time(
        curves_dir=curves_dir,
        instance_tag=INSTANCE_TAG,
        n_processes=N_processes,
        approx_time=approx_time,
    )

    time_grid = np.asarray(curve_data["time_grid"], dtype=np.float64)
    T = len(time_grid)

    print()
    print("Loaded setup")
    print("------------")
    print("simID =", simID)
    print("N =", N)
    print("E =", E)
    print("number of full process files =", len(full_process_paths))
    print("number of curve times =", T)
    print("curve_path =", curve_path)
    print("matched_curve_time =", matched_curve_time)
    print("n_workers =", n_workers)
    print("common neighbour entries =", len(common_nodes_flat))

    # ============================================================
    # combined parallel PASS:
    # snapshot averages + PASS 1 local covariance ingredients
    # ============================================================

    n_chunks = max(1, n_workers * n_chunks_multiplier)
    chunks = make_chunks(full_process_paths, n_chunks)

    print()
    print("Combined snapshot/PASS 1")
    print("------------------------")
    print("number of chunks =", len(chunks))

    sum_S = np.zeros(T, dtype=np.float64)
    sum_S2 = np.zeros(T, dtype=np.float64)
    sum_SI = np.zeros(T, dtype=np.float64)
    sum_S_SI = np.zeros(T, dtype=np.float64)

    sum_X = np.zeros((T, N), dtype=np.float64)
    sum_S_SI_local = np.zeros(T, dtype=np.float64)

    n_loaded_all = np.zeros(T, dtype=np.int64)

    t_compute_start = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(compute_snapshot_and_local_chunk, chunk)
            for chunk in chunks
        ]

        for done_idx, future in enumerate(as_completed(futures), start=1):
            (
                local_sum_S,
                local_sum_S2,
                local_sum_SI,
                local_sum_S_SI,
                local_sum_X,
                local_sum_S_SI_local,
                local_n,
            ) = future.result()

            sum_S += local_sum_S
            sum_S2 += local_sum_S2
            sum_SI += local_sum_SI
            sum_S_SI += local_sum_S_SI

            sum_X += local_sum_X
            sum_S_SI_local += local_sum_S_SI_local

            n_loaded_all += local_n

            if done_idx % 5 == 0 or done_idx == len(chunks):
                print(f"Combined PASS 1 finished {done_idx}/{len(chunks)} chunks")

    t_compute_end = time.time()

    print()
    print("Finished combined snapshot/PASS 1.")
    print("Computation time =", t_compute_end - t_compute_start, "seconds")
    print("Computation time =", (t_compute_end - t_compute_start) / 60, "minutes")

    # ============================================================
    # convert sums to averages
    # ============================================================

    valid = n_loaded_all > 0

    mean_S_all = np.full(T, np.nan, dtype=np.float64)
    mean_S2_all = np.full(T, np.nan, dtype=np.float64)
    mean_SI_all = np.full(T, np.nan, dtype=np.float64)
    mean_S_SI_all = np.full(T, np.nan, dtype=np.float64)

    mean_X = np.full((T, N), np.nan, dtype=np.float64)
    mean_S_SI_local_all = np.full(T, np.nan, dtype=np.float64)

    mean_S_all[valid] = sum_S[valid] / n_loaded_all[valid]
    mean_S2_all[valid] = sum_S2[valid] / n_loaded_all[valid]
    mean_SI_all[valid] = sum_SI[valid] / n_loaded_all[valid]
    mean_S_SI_all[valid] = sum_S_SI[valid] / n_loaded_all[valid]

    mean_X[valid] = sum_X[valid] / n_loaded_all[valid, None]
    mean_S_SI_local_all[valid] = sum_S_SI_local[valid] / n_loaded_all[valid]

    var_S_snapshot_all = mean_S2_all - mean_S_all**2
    cov_S_SI_all = mean_S_SI_all - mean_S_all * mean_SI_all

    dVarS_dt_snapshot_all = (
        beta * mean_SI_all
        - 2.0 * beta * cov_S_SI_all
    )

    print()
    print("Loaded realizations per time")
    print("----------------------------")
    print("min =", np.min(n_loaded_all))
    print("max =", np.max(n_loaded_all))
    print("expected =", len(full_process_paths))

    # ============================================================
    # curve-file variance and numerical derivative
    # ============================================================

    var_S_curve_all = curve_data["var_fractions"][:, 0] * N**2
    var_S_curve_corrected_all = var_S_curve_all * (N - 1) / N

    dVarS_dt_curve_all = np.gradient(
        var_S_curve_all,
        time_grid,
    )

    dVarS_dt_curve_corrected_all = np.gradient(
        var_S_curve_corrected_all,
        time_grid,
    )

    dVarS_dt_snapshot_numerical_all = np.gradient(
        var_S_snapshot_all,
        time_grid,
    )

    # ============================================================
    # integrate full covariance formula
    # ============================================================

    dt = np.diff(time_grid)

    integral_dVarS_dt_snapshot = np.zeros_like(dVarS_dt_snapshot_all)

    integral_dVarS_dt_snapshot[1:] = np.cumsum(
        0.5
        * (
            dVarS_dt_snapshot_all[1:]
            + dVarS_dt_snapshot_all[:-1]
        )
        * dt
    )

    var_S_integrated_from_snapshot_formula = (
        var_S_snapshot_all[0]
        + integral_dVarS_dt_snapshot
    )

    # ============================================================
    # save exact integrated covariance-formula curve
    # ============================================================

    save_dir_integrated = base_dir / "IntegratedCovarianceFormula"
    save_dir_integrated.mkdir(parents=True, exist_ok=True)

    save_path = save_dir_integrated / (
        f"VarS_from_integrated_covariance_formula_"
        f"{simID}_{INSTANCE_TAG}_Nproc{N_processes}_beta{beta:g}.npz"
    )

    np.savez_compressed(
        save_path,

        # metadata
        simID=simID,
        instance_number=INSTANCE_NUMBER,
        instance_tag=INSTANCE_TAG,
        N=N,
        N_processes=N_processes,
        beta=beta,

        # time grid
        time_grid=time_grid,

        # exact integrated covariance-formula curve
        var_S_integrated_from_snapshot_formula=var_S_integrated_from_snapshot_formula,
        integral_dVarS_dt_snapshot=integral_dVarS_dt_snapshot,

        # ingredients
        dVarS_dt_snapshot_all=dVarS_dt_snapshot_all,
        cov_S_SI_all=cov_S_SI_all,
        mean_SI_all=mean_SI_all,
        mean_S_all=mean_S_all,
        mean_S2_all=mean_S2_all,
        mean_S_SI_all=mean_S_SI_all,

        # initial condition
        var_S_initial=var_S_snapshot_all[0],

        # comparison curves
        var_S_snapshot_all=var_S_snapshot_all,
        var_S_curve_all=var_S_curve_all,
        var_S_curve_corrected_all=var_S_curve_corrected_all,

        # diagnostics
        n_loaded_all=n_loaded_all,
    )

    print()
    print("Saved integrated covariance-formula Var(S) curve to:")
    print(save_path)

    # ============================================================
    # precompute mean local susceptible around edge
    # used in PASS 2
    # ============================================================

    print()
    print("Precomputing local_mean_S_around_edge_all")
    print("-----------------------------------------")

    t_precompute_start = time.time()

    local_mean_S_around_edge_all = np.zeros((T, E), dtype=np.float64)

    for tidx in range(T):
        if valid[tidx]:
            local_mean_S_around_edge_all[tidx] = closed_edge_local_sum(mean_X[tidx])

        if tidx % 10 == 0 or tidx == T - 1:
            print(f"precomputed {tidx + 1}/{T}")

    t_precompute_end = time.time()

    print("Precompute time =", t_precompute_end - t_precompute_start, "seconds")
    print("Precompute time =", (t_precompute_end - t_precompute_start) / 60, "minutes")

    # ============================================================
    # parallel PASS 2:
    # compute {<S><[SI]>}_L
    # ============================================================

    print()
    print("PASS 2")
    print("------")
    print("number of chunks =", len(chunks))

    sum_factorized_local = np.zeros(T, dtype=np.float64)
    sum_factorized_S_on_I = np.zeros(T, dtype=np.float64)
    n_loaded_factorized = np.zeros(T, dtype=np.int64)

    t_compute_start = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(compute_factorized_chunk, chunk)
            for chunk in chunks
        ]

        for done_idx, future in enumerate(as_completed(futures), start=1):
            local_factorized, local_S_on_I, local_n = future.result()

            sum_factorized_local += local_factorized
            sum_factorized_S_on_I += local_S_on_I
            n_loaded_factorized += local_n

            if done_idx % 5 == 0 or done_idx == len(chunks):
                print(f"PASS 2 finished {done_idx}/{len(chunks)} chunks")

    t_compute_end = time.time()

    print()
    print("Finished PASS 2.")
    print("Computation time =", t_compute_end - t_compute_start, "seconds")
    print("Computation time =", (t_compute_end - t_compute_start) / 60, "minutes")

    valid_factorized = n_loaded_factorized > 0

    factorized_local_all = np.full(T, np.nan, dtype=np.float64)
    factorized_S_on_I_all = np.full(T, np.nan, dtype=np.float64)

    factorized_local_all[valid_factorized] = (
        sum_factorized_local[valid_factorized]
        / n_loaded_factorized[valid_factorized]
    )

    factorized_S_on_I_all[valid_factorized] = (
        sum_factorized_S_on_I[valid_factorized]
        / n_loaded_factorized[valid_factorized]
    )

    cov_S_SI_local_all = mean_S_SI_local_all - factorized_local_all

    # term (2) contribution to the covariance itself is negative:
    cov_S_on_I_contribution_all = -factorized_S_on_I_all

    dVarS_dt_localcov_all = (
        beta * mean_SI_all
        - 2.0 * beta * cov_S_SI_local_all
    )

    # ============================================================
    # integrate local-covariance derivative
    # ============================================================

    integral_dVarS_dt_localcov = np.zeros_like(dVarS_dt_localcov_all)

    integral_dVarS_dt_localcov[1:] = np.cumsum(
        0.5
        * (
            dVarS_dt_localcov_all[1:]
            + dVarS_dt_localcov_all[:-1]
        )
        * dt
    )

    var_S_integrated_from_localcov_formula = (
        var_S_snapshot_all[0]
        + integral_dVarS_dt_localcov
    )

    dVarS_dt_integrated_full_numerical_all = np.gradient(
        var_S_integrated_from_snapshot_formula,
        time_grid,
    )

    dVarS_dt_integrated_local_numerical_all = np.gradient(
        var_S_integrated_from_localcov_formula,
        time_grid,
    )

    # ============================================================
    # Closure 1 integrated variance from curve-file skm/ikm data
    # ============================================================

    mean_skm_closure_1 = curve_data["mean_skm"]
    mean_ikm_closure_1 = curve_data["mean_ikm"]

    deg_closure_1 = np.zeros(N, dtype=np.int64)
    np.add.at(deg_closure_1, v1, 1)
    np.add.at(deg_closure_1, v2, 1)

    K_closure_1 = mean_skm_closure_1.shape[1]
    M_closure_1 = mean_skm_closure_1.shape[2]

    n_k_closure_1 = np.bincount(
        deg_closure_1.astype(int),
        minlength=K_closure_1,
    ).astype(np.float64)

    if len(n_k_closure_1) > K_closure_1:
        raise ValueError(
            "Some graph degrees exceed the k-axis stored in mean_skm/mean_ikm."
        )

    n_k_closure_1 = n_k_closure_1[:K_closure_1]
    Pk_closure_1 = n_k_closure_1 / N

    # phi_S_m_if_k[t,k,m] = P(S and m infected neighbours | degree k)
    # phi_I_m_if_k[t,k,m] = P(I and m infected neighbours | degree k)

    phi_S_m_if_k_closure_1 = np.zeros_like(mean_skm_closure_1, dtype=np.float64)
    phi_I_m_if_k_closure_1 = np.zeros_like(mean_ikm_closure_1, dtype=np.float64)

    np.divide(
        mean_skm_closure_1,
        Pk_closure_1[None, :, None],
        out=phi_S_m_if_k_closure_1,
        where=(Pk_closure_1[None, :, None] > 0),
    )

    np.divide(
        mean_ikm_closure_1,
        Pk_closure_1[None, :, None],
        out=phi_I_m_if_k_closure_1,
        where=(Pk_closure_1[None, :, None] > 0),
    )

    k_values_full_closure_1 = np.arange(K_closure_1, dtype=np.float64)
    m_values_closure_1 = np.arange(M_closure_1, dtype=np.float64)

    k_grid_closure_1 = k_values_full_closure_1[None, :, None]
    m_grid_closure_1 = m_values_closure_1[None, None, :]

    mbar_grid_closure_1 = k_grid_closure_1 - m_grid_closure_1
    mbar_grid_closure_1 = np.where(mbar_grid_closure_1 >= 0, mbar_grid_closure_1, 0.0)

    # ============================================================
    # SI, SSI, SIS from skm/ikm
    # ============================================================

    SI_closure_1 = np.sum(
        n_k_closure_1[None, :, None]
        * phi_S_m_if_k_closure_1
        * m_grid_closure_1,
        axis=(1, 2),
    )

    SSI_closure_1 = np.sum(
        n_k_closure_1[None, :, None]
        * phi_S_m_if_k_closure_1
        * m_grid_closure_1
        * mbar_grid_closure_1,
        axis=(1, 2),
    )

    binom_mbar_2_grid_closure_1 = np.where(
        mbar_grid_closure_1 >= 2,
        mbar_grid_closure_1 * (mbar_grid_closure_1 - 1.0) / 2.0,
        0.0,
    )

    SIS_closure_1 = np.sum(
        n_k_closure_1[None, :, None]
        * phi_I_m_if_k_closure_1
        * binom_mbar_2_grid_closure_1,
        axis=(1, 2),
    )

    # ============================================================
    # Closure 1 product approximation
    # ============================================================

    phi_S_if_k_closure_1 = np.sum(
        phi_S_m_if_k_closure_1,
        axis=2,
    )

    total_edges_closure_1 = len(v1)

    phi_S_degree_weighted_closure_1 = np.sum(
        n_k_closure_1[None, :]
        * k_values_full_closure_1[None, :]
        * phi_S_if_k_closure_1,
        axis=1,
    ) / (2.0 * total_edges_closure_1)

    IS_k_closure_1 = np.sum(
        n_k_closure_1[None, :, None]
        * m_grid_closure_1
        * phi_S_m_if_k_closure_1,
        axis=2,
    )

    SI_k_closure_1 = np.sum(
        n_k_closure_1[None, :, None]
        * mbar_grid_closure_1
        * phi_I_m_if_k_closure_1,
        axis=2,
    )

    k_minus_1_grid_closure_1 = np.maximum(
        k_values_full_closure_1[None, :] - 1.0,
        0.0,
    )

    local_product_term_1_closure_1 = np.sum(
        IS_k_closure_1 * phi_S_if_k_closure_1,
        axis=1,
    )

    local_product_term_2_closure_1 = np.sum(
        SI_k_closure_1 * phi_S_if_k_closure_1,
        axis=1,
    )

    local_product_term_3_closure_1 = (
        np.sum(
            IS_k_closure_1 * k_minus_1_grid_closure_1,
            axis=1,
        )
        * phi_S_degree_weighted_closure_1
    )

    local_product_term_4_closure_1 = (
        np.sum(
            SI_k_closure_1 * k_minus_1_grid_closure_1,
            axis=1,
        )
        * phi_S_degree_weighted_closure_1
    )

    E_of_S_E_of_SI_LOCAL_no_triangles_closure_1 = (
        local_product_term_1_closure_1
        + local_product_term_2_closure_1
        + local_product_term_3_closure_1
        + local_product_term_4_closure_1
    )

    # ============================================================
    # dVar(S)/dt closure 1, no triangles
    # ============================================================

    dVarS_dt_no_triangles_closure_1 = (
        2.0 * beta * E_of_S_E_of_SI_LOCAL_no_triangles_closure_1
        - beta * SI_closure_1
        - 4.0 * beta * SIS_closure_1
        - 2.0 * beta * SSI_closure_1
    )

    # ============================================================
    # integrate Closure 1 derivative
    # ============================================================

    integral_dVarS_dt_closure_1 = np.zeros_like(
        dVarS_dt_no_triangles_closure_1
    )

    integral_dVarS_dt_closure_1[1:] = np.cumsum(
        0.5
        * (
            dVarS_dt_no_triangles_closure_1[1:]
            + dVarS_dt_no_triangles_closure_1[:-1]
        )
        * dt
    )

    var_S_integrated_from_closure_1 = (
        var_S_curve_corrected_all[0]
        + integral_dVarS_dt_closure_1
    )

    dVarS_dt_integrated_closure_1_numerical_all = np.gradient(
        var_S_integrated_from_closure_1,
        time_grid,
    )

    # ============================================================
    # save all comparison curves
    # ============================================================

    save_dir = base_dir / "VarianceComparisons"
    save_dir.mkdir(parents=True, exist_ok=True)

    fig_dir = save_dir / "Figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    save_path_compare = save_dir / (
        f"Variance_comparison_full_vs_local_covariance_"
        f"{simID}_{INSTANCE_TAG}_Nproc{N_processes}_beta{beta:g}.npz"
    )

    np.savez_compressed(
        save_path_compare,

        simID=simID,
        instance_number=INSTANCE_NUMBER,
        instance_tag=INSTANCE_TAG,
        N=N,
        E=E,
        N_processes=N_processes,
        beta=beta,

        graph_path=str(graph_path),
        curve_path=str(curve_path),
        matched_curve_time=matched_curve_time,

        time_grid=time_grid,

        var_S_snapshot_all=var_S_snapshot_all,
        var_S_curve_all=var_S_curve_all,
        var_S_curve_corrected_all=var_S_curve_corrected_all,

        var_S_integrated_from_snapshot_formula=var_S_integrated_from_snapshot_formula,
        var_S_integrated_from_localcov_formula=var_S_integrated_from_localcov_formula,

        integral_dVarS_dt_snapshot=integral_dVarS_dt_snapshot,
        integral_dVarS_dt_localcov=integral_dVarS_dt_localcov,

        # Closure 1 integrated variance
        var_S_integrated_from_closure_1=var_S_integrated_from_closure_1,
        integral_dVarS_dt_closure_1=integral_dVarS_dt_closure_1,
        dVarS_dt_no_triangles_closure_1=dVarS_dt_no_triangles_closure_1,
        dVarS_dt_integrated_closure_1_numerical_all=dVarS_dt_integrated_closure_1_numerical_all,

        # Closure 1 ingredients
        SI_closure_1=SI_closure_1,
        SSI_closure_1=SSI_closure_1,
        SIS_closure_1=SIS_closure_1,
        E_of_S_E_of_SI_LOCAL_no_triangles_closure_1=E_of_S_E_of_SI_LOCAL_no_triangles_closure_1,

        # Closure 1 product terms
        local_product_term_1_closure_1=local_product_term_1_closure_1,
        local_product_term_2_closure_1=local_product_term_2_closure_1,
        local_product_term_3_closure_1=local_product_term_3_closure_1,
        local_product_term_4_closure_1=local_product_term_4_closure_1,

        dVarS_dt_snapshot_numerical_all=dVarS_dt_snapshot_numerical_all,
        dVarS_dt_curve_all=dVarS_dt_curve_all,
        dVarS_dt_curve_corrected_all=dVarS_dt_curve_corrected_all,
        dVarS_dt_snapshot_all=dVarS_dt_snapshot_all,
        dVarS_dt_localcov_all=dVarS_dt_localcov_all,

        dVarS_dt_integrated_full_numerical_all=dVarS_dt_integrated_full_numerical_all,
        dVarS_dt_integrated_local_numerical_all=dVarS_dt_integrated_local_numerical_all,

        cov_S_SI_all=cov_S_SI_all,
        cov_S_SI_local_all=cov_S_SI_local_all,
        cov_S_on_I_contribution_all=cov_S_on_I_contribution_all,

        mean_SI_all=mean_SI_all,
        mean_S_all=mean_S_all,
        mean_S2_all=mean_S2_all,
        mean_S_SI_all=mean_S_SI_all,
        mean_S_SI_local_all=mean_S_SI_local_all,
        factorized_local_all=factorized_local_all,
        factorized_S_on_I_all=factorized_S_on_I_all,

        n_loaded_all=n_loaded_all,
        n_loaded_factorized=n_loaded_factorized,
    )

    print()
    print("Saved full/local/Closure 1 variance comparison curves to:")
    print(save_path_compare)

    # ============================================================
    # derivative comparison plot
    # ============================================================

    fig = plt.figure(figsize=(9, 5.5))

    plt.plot(
        time_grid,
        dVarS_dt_snapshot_numerical_all,
        linewidth=2,
        label=r"numerical derivative of snapshot $\mathrm{Var}(S)$",
    )

    plt.plot(
        time_grid,
        dVarS_dt_curve_corrected_all,
        linestyle="--",
        linewidth=2,
        label=r"numerical derivative of curve-file $\mathrm{Var}(S)$",
    )

    plt.plot(
        time_grid,
        dVarS_dt_snapshot_all,
        linestyle="-.",
        linewidth=2,
        label=r"full covariance RHS",
    )

    plt.plot(
        time_grid,
        dVarS_dt_localcov_all,
        linestyle=":",
        linewidth=2.5,
        label=r"local covariance RHS",
    )

    plt.plot(
        time_grid,
        dVarS_dt_no_triangles_closure_1,
        linestyle=(0, (3, 1, 1, 1)),
        linewidth=2.5,
        label=r"Closure 1 RHS",
    )

    plt.axhline(0.0, color="black", linewidth=1, linestyle=":")

    plt.xlabel(r"$t$")
    plt.ylabel(r"$d\mathrm{Var}(S)/dt$")
    plt.title(r"Derivative comparison for $\mathrm{Var}(S)$")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_and_show(
        fig,
        fig_dir / (
            f"Derivative_comparison_VarS_"
            f"{simID}_{INSTANCE_TAG}_Nproc{N_processes}_beta{beta:g}.png"
        ),
    )

    # ============================================================
    # integrated variance comparison plot
    # ============================================================

    fig = plt.figure(figsize=(9, 5.5))

    plt.plot(
        time_grid,
        var_S_snapshot_all,
        linewidth=2,
        label=r"snapshot $\mathrm{Var}(S)$",
    )

    plt.plot(
        time_grid,
        var_S_curve_corrected_all,
        linestyle="--",
        linewidth=2,
        label=r"curve-file $\mathrm{Var}(S)$",
    )

    plt.plot(
        time_grid,
        var_S_integrated_from_snapshot_formula,
        linestyle="-.",
        linewidth=2,
        label=r"integrated full covariance formula",
    )

    plt.plot(
        time_grid,
        var_S_integrated_from_localcov_formula,
        linestyle=":",
        linewidth=2.5,
        label=r"integrated local covariance formula",
    )

    plt.plot(
        time_grid,
        var_S_integrated_from_closure_1,
        linestyle=(0, (3, 1, 1, 1)),
        linewidth=2.5,
        label=r"integrated Closure 1 formula",
    )

    plt.xlabel(r"$t$")
    plt.ylabel(r"$\mathrm{Var}(S)$")
    plt.title(r"Variance comparison: direct, curve file, covariance formulas, Closure 1")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_and_show(
        fig,
        fig_dir / (
            f"Integrated_variance_comparison_"
            f"{simID}_{INSTANCE_TAG}_Nproc{N_processes}_beta{beta:g}.png"
        ),
    )

    # ============================================================
    # error plot: relative to snapshot variance
    # ============================================================

    fig = plt.figure(figsize=(9, 5.5))

    plt.plot(
        time_grid,
        var_S_curve_corrected_all - var_S_snapshot_all,
        linewidth=2,
        label=r"curve file $-$ snapshot",
    )

    plt.plot(
        time_grid,
        var_S_integrated_from_snapshot_formula - var_S_snapshot_all,
        linestyle="--",
        linewidth=2,
        label=r"integrated full covariance $-$ snapshot",
    )

    plt.plot(
        time_grid,
        var_S_integrated_from_localcov_formula - var_S_snapshot_all,
        linestyle=":",
        linewidth=2.5,
        label=r"integrated local covariance $-$ snapshot",
    )

    plt.plot(
        time_grid,
        var_S_integrated_from_closure_1 - var_S_snapshot_all,
        linestyle=(0, (3, 1, 1, 1)),
        linewidth=2.5,
        label=r"integrated Closure 1 $-$ snapshot",
    )

    plt.axhline(0.0, color="black", linewidth=1, linestyle=":")

    plt.xlabel(r"$t$")
    plt.ylabel(r"error")
    plt.title(r"Variance errors relative to snapshot $\mathrm{Var}(S)$")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_and_show(
        fig,
        fig_dir / (
            f"Variance_errors_relative_to_snapshot_"
            f"{simID}_{INSTANCE_TAG}_Nproc{N_processes}_beta{beta:g}.png"
        ),
    )

    # ============================================================
    # error plot: relative to curve-file variance
    # ============================================================

    fig = plt.figure(figsize=(9, 5.5))

    plt.plot(
        time_grid,
        var_S_snapshot_all - var_S_curve_corrected_all,
        linewidth=2,
        label=r"snapshot $-$ curve file",
    )

    plt.plot(
        time_grid,
        var_S_integrated_from_snapshot_formula - var_S_curve_corrected_all,
        linestyle="--",
        linewidth=2,
        label=r"integrated full covariance $-$ curve file",
    )

    plt.plot(
        time_grid,
        var_S_integrated_from_localcov_formula - var_S_curve_corrected_all,
        linestyle=":",
        linewidth=2.5,
        label=r"integrated local covariance $-$ curve file",
    )

    plt.plot(
        time_grid,
        var_S_integrated_from_closure_1 - var_S_curve_corrected_all,
        linestyle=(0, (3, 1, 1, 1)),
        linewidth=2.5,
        label=r"integrated Closure 1 $-$ curve file",
    )

    plt.axhline(0.0, color="black", linewidth=1, linestyle=":")

    plt.xlabel(r"$t$")
    plt.ylabel(r"error")
    plt.title(r"Variance errors relative to curve-file $\mathrm{Var}(S)$")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_and_show(
        fig,
        fig_dir / (
            f"Variance_errors_relative_to_curve_file_"
            f"{simID}_{INSTANCE_TAG}_Nproc{N_processes}_beta{beta:g}.png"
        ),
    )

    # ============================================================
    # diagnostics
    # ============================================================

    t_total_end = time.time()

    print()
    print("Finished everything.")
    print("Total time =", t_total_end - t_total_start, "seconds")
    print("Total time =", (t_total_end - t_total_start) / 60, "minutes")

    print()
    print("Saved comparison npz:")
    print(save_path_compare)

    if SAVE_FIGURES:
        print()
        print("Saved figures in:")
        print(fig_dir)


if __name__ == "__main__":
    main()