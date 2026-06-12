import numpy as np
from pathlib import Path
import time
import matplotlib.pyplot as plt

from concurrent.futures import ThreadPoolExecutor, as_completed

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

n_workers = 30

t_total_start = time.time()

# ============================================================
# load graph
# ============================================================

graph_path, N, v1, v2 = load_graph_file(
    graphs_dir=graphs_dir,
    instance_tag=INSTANCE_TAG,
)


# ============================================================
# local motif helper for Cov(S,[SI])_L
# ============================================================

v1 = np.asarray(v1, dtype=np.int64)
v2 = np.asarray(v2, dtype=np.int64)
E = len(v1)

neighbors = [set() for _ in range(N)]
for a, b in zip(v1, v2):
    neighbors[a].add(b)
    neighbors[b].add(a)

# common neighbors are needed so triangle nodes are not double-counted
common_neighbors_by_edge = [
    np.asarray(sorted(neighbors[a] & neighbors[b]), dtype=np.int64)
    for a, b in zip(v1, v2)
]


def closed_edge_local_sum(x):
    """
    For every edge e=(a,b), return

        sum_{i in {a,b} union N(a) union N(b)} x_i

    This is exactly the local set:
        i = j,
        i = ell,
        i neighbor of j,
        i neighbor of ell,

    with duplicates removed.
    """

    x = np.asarray(x, dtype=np.float64)

    nbr_sum = np.zeros(N, dtype=np.float64)

    # add x_b to node a's neighbor sum, and x_a to node b's neighbor sum
    np.add.at(nbr_sum, v1, x[v2])
    np.add.at(nbr_sum, v2, x[v1])

    common_sum = np.fromiter(
        (x[common].sum() for common in common_neighbors_by_edge),
        dtype=np.float64,
        count=E,
    )

    # For edge (a,b):
    # closed neighborhood union = N[a] union N[b]
    # Because a and b are adjacent, this equals:
    # sum_N(a) + sum_N(b) - sum_common_neighbors
    return nbr_sum[v1] + nbr_sum[v2] - common_sum

# ============================================================
# find full process paths and load curve file
# ============================================================

full_process_paths = find_full_process_paths(
    full_process_dir=full_process_dir,
    instance_tag=INSTANCE_TAG,
    n_processes=N_processes,
)

curve_path, curve_idx, matched_curve_time, curve_data = load_curve_and_match_time(
    curves_dir=curves_dir,
    instance_tag=INSTANCE_TAG,
    n_processes=N_processes,
    approx_time=approx_time,
)

time_grid = curve_data["time_grid"]
T = len(time_grid)

print()
print("Loaded setup")
print("------------")
print("simID =", simID)
print("N =", N)
print("number of full process files =", len(full_process_paths))
print("number of curve times =", T)
print("curve_path =", curve_path)


# ============================================================
# helper: load one full-process file
# ============================================================

def load_full_process_file(path):
    """
    Loads one full-process .npz file.

    Your files use:
        raw_times
        raw_vertex_states
    """

    d = np.load(path, allow_pickle=True)

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


# ============================================================
# helper: compute all curve times for one realization
# ============================================================

def compute_one_realization(path):
    times, states_all = load_full_process_file(path)

    local_sum_S = np.zeros(T, dtype=np.float64)
    local_sum_S2 = np.zeros(T, dtype=np.float64)
    local_sum_SI = np.zeros(T, dtype=np.float64)
    local_sum_S_SI = np.zeros(T, dtype=np.float64)
    local_n = np.zeros(T, dtype=np.int64)

    # For each curve time, use latest snapshot at or before that time
    snapshot_indices = np.searchsorted(times, time_grid, side="right") - 1

    for tidx, snap_idx in enumerate(snapshot_indices):
        if snap_idx < 0:
            continue

        states = states_all[snap_idx]

        S_count = np.sum(states == S_state)

        SI_count = np.sum(
            (
                (states[v1] == S_state)
                & (states[v2] == I_state)
            )
            |
            (
                (states[v1] == I_state)
                & (states[v2] == S_state)
            )
        )

        local_sum_S[tidx] = S_count
        local_sum_S2[tidx] = S_count**2
        local_sum_SI[tidx] = SI_count
        local_sum_S_SI[tidx] = S_count * SI_count
        local_n[tidx] = 1

    return local_sum_S, local_sum_S2, local_sum_SI, local_sum_S_SI, local_n


# ============================================================
# compute sums in parallel over realizations
# ============================================================

sum_S = np.zeros(T, dtype=np.float64)
sum_S2 = np.zeros(T, dtype=np.float64)
sum_SI = np.zeros(T, dtype=np.float64)
sum_S_SI = np.zeros(T, dtype=np.float64)
n_loaded_all = np.zeros(T, dtype=np.int64)

t_compute_start = time.time()

with ThreadPoolExecutor(max_workers=n_workers) as executor:
    futures = [
        executor.submit(compute_one_realization, path)
        for path in full_process_paths
    ]

    for done_idx, future in enumerate(as_completed(futures), start=1):
        local_sum_S, local_sum_S2, local_sum_SI, local_sum_S_SI, local_n = future.result()

        sum_S += local_sum_S
        sum_S2 += local_sum_S2
        sum_SI += local_sum_SI
        sum_S_SI += local_sum_S_SI
        n_loaded_all += local_n

        if done_idx % 50 == 0 or done_idx == len(full_process_paths):
            print(f"Finished {done_idx}/{len(full_process_paths)} realizations")

t_compute_end = time.time()

print()
print("Finished computing snapshot averages.")
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

mean_S_all[valid] = sum_S[valid] / n_loaded_all[valid]
mean_S2_all[valid] = sum_S2[valid] / n_loaded_all[valid]
mean_SI_all[valid] = sum_SI[valid] / n_loaded_all[valid]
mean_S_SI_all[valid] = sum_S_SI[valid] / n_loaded_all[valid]

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

dVarS_dt_curve_all = np.gradient(
    var_S_curve_all*(N-1)/N,
    time_grid
)

# ============================================================
# integrate covariance formula
# ============================================================

integral_dVarS_dt_snapshot = np.zeros_like(dVarS_dt_snapshot_all)

dt = np.diff(time_grid)

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

save_dir = base_dir / "IntegratedCovarianceFormula"
save_dir.mkdir(parents=True, exist_ok=True)

save_path = save_dir / (
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

    # diagnostics
    n_loaded_all=n_loaded_all,
)

print()
print("Saved integrated covariance-formula Var(S) curve to:")
print(save_path)

# ============================================================
# PASS 1:
# collect <X_i>, <[SI]>, and <S[SI]>_L
# ============================================================

sum_X = np.zeros((T, N), dtype=np.float64)

sum_SI = np.zeros(T, dtype=np.float64)
sum_S_SI_local = np.zeros(T, dtype=np.float64)

n_loaded_all = np.zeros(T, dtype=np.int64)

t_compute_start = time.time()

for done_idx, path in enumerate(full_process_paths, start=1):
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

        # total [SI]
        SI_count = np.sum(edge_is_SI)

        # local number of susceptible nodes around each SI edge
        local_S_around_edge = closed_edge_local_sum(S)

        # this is (S[SI])_L for this realization
        S_SI_local = np.sum(edge_is_SI * local_S_around_edge)

        sum_X[tidx] += S
        sum_SI[tidx] += SI_count
        sum_S_SI_local[tidx] += S_SI_local
        n_loaded_all[tidx] += 1

    if done_idx % 50 == 0 or done_idx == len(full_process_paths):
        print(f"PASS 1 finished {done_idx}/{len(full_process_paths)} realizations")

valid = n_loaded_all > 0

mean_X = np.full((T, N), np.nan, dtype=np.float64)
mean_SI_all = np.full(T, np.nan, dtype=np.float64)
mean_S_SI_local_all = np.full(T, np.nan, dtype=np.float64)

mean_X[valid] = sum_X[valid] / n_loaded_all[valid, None]
mean_SI_all[valid] = sum_SI[valid] / n_loaded_all[valid]
mean_S_SI_local_all[valid] = sum_S_SI_local[valid] / n_loaded_all[valid]

# ============================================================
# PASS 2:
# compute {<S><[SI]>}_L
# ============================================================

sum_factorized_local = np.zeros(T, dtype=np.float64)

# diagnostic: this is the i = ell, "S on I" piece
sum_factorized_S_on_I = np.zeros(T, dtype=np.float64)

n_loaded_factorized = np.zeros(T, dtype=np.int64)

for done_idx, path in enumerate(full_process_paths, start=1):
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

        # local factorized piece:
        # sum_edges <X_j(1-X_l)> * sum_{local i} <X_i>
        local_mean_S_around_edge = closed_edge_local_sum(mean_X[tidx])

        sum_factorized_local[tidx] += np.sum(
            edge_is_SI * local_mean_S_around_edge
        )

        # Diagnostic only:
        # term (2), i = ell, "S on I"
        # If v1 is S and v2 is I, then ell = v2.
        # If v2 is S and v1 is I, then ell = v1.
        sum_factorized_S_on_I[tidx] += (
            np.sum(edge_v1_S_v2_I * mean_X[tidx, v2])
            + np.sum(edge_v2_S_v1_I * mean_X[tidx, v1])
        )

        n_loaded_factorized[tidx] += 1

    if done_idx % 50 == 0 or done_idx == len(full_process_paths):
        print(f"PASS 2 finished {done_idx}/{len(full_process_paths)} realizations")

factorized_local_all = np.full(T, np.nan, dtype=np.float64)
factorized_S_on_I_all = np.full(T, np.nan, dtype=np.float64)

factorized_local_all[valid] = (
    sum_factorized_local[valid] / n_loaded_factorized[valid]
)

factorized_S_on_I_all[valid] = (
    sum_factorized_S_on_I[valid] / n_loaded_factorized[valid]
)

cov_S_SI_local_all = mean_S_SI_local_all - factorized_local_all

# term (2) contribution to the covariance itself is negative:
cov_S_on_I_contribution_all = -factorized_S_on_I_all

dVarS_dt_localcov_all = (
    beta * mean_SI_all
    - 2.0 * beta * cov_S_SI_local_all
)


# ============================================================
# compare all variance reconstructions
# ============================================================

# ddof = 0 vs ddof = 2 (population vs unbiased)
var_S_curve_corrected_all = var_S_curve_all * (N - 1) / N

# Numerical derivatives of directly measured variances
dVarS_dt_snapshot_numerical_all = np.gradient(
    var_S_snapshot_all,
    time_grid,
)

dVarS_dt_curve_corrected_all = np.gradient(
    var_S_curve_corrected_all,
    time_grid,
)

# ============================================================
# integrate local-covariance derivative
# ============================================================

dt = np.diff(time_grid)

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

# Optional: numerical derivatives of the integrated curves
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

# degree vector
deg_closure_1 = np.zeros(N, dtype=np.int64)
np.add.at(deg_closure_1, v1, 1)
np.add.at(deg_closure_1, v2, 1)

K_closure_1 = mean_skm_closure_1.shape[1]
M_closure_1 = mean_skm_closure_1.shape[2]

n_k_closure_1 = np.bincount(
    deg_closure_1.astype(int),
    minlength=K_closure_1
).astype(np.float64)

if len(n_k_closure_1) > K_closure_1:
    raise ValueError(
        "Some graph degrees exceed the k-axis stored in mean_skm/mean_ikm."
    )

n_k_closure_1 = n_k_closure_1[:K_closure_1]

Pk_closure_1 = n_k_closure_1 / N

# phi_S_m_if_k[t,k,m] = P(S and m infected neighbors | degree k)
# phi_I_m_if_k[t,k,m] = P(I and m infected neighbors | degree k)

phi_S_m_if_k_closure_1 = np.zeros_like(mean_skm_closure_1, dtype=np.float64)
phi_I_m_if_k_closure_1 = np.zeros_like(mean_ikm_closure_1, dtype=np.float64)

np.divide(
    mean_skm_closure_1,
    Pk_closure_1[None, :, None],
    out=phi_S_m_if_k_closure_1,
    where=(Pk_closure_1[None, :, None] > 0)
)

np.divide(
    mean_ikm_closure_1,
    Pk_closure_1[None, :, None],
    out=phi_I_m_if_k_closure_1,
    where=(Pk_closure_1[None, :, None] > 0)
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
    axis=(1, 2)
)

SSI_closure_1 = np.sum(
    n_k_closure_1[None, :, None]
    * phi_S_m_if_k_closure_1
    * m_grid_closure_1
    * mbar_grid_closure_1,
    axis=(1, 2)
)

binom_mbar_2_grid_closure_1 = np.where(
    mbar_grid_closure_1 >= 2,
    mbar_grid_closure_1 * (mbar_grid_closure_1 - 1.0) / 2.0,
    0.0
)

SIS_closure_1 = np.sum(
    n_k_closure_1[None, :, None]
    * phi_I_m_if_k_closure_1
    * binom_mbar_2_grid_closure_1,
    axis=(1, 2)
)

# ============================================================
# Closure 1 product approximation
# ============================================================

phi_S_if_k_closure_1 = np.sum(
    phi_S_m_if_k_closure_1,
    axis=2
)

total_edges_closure_1 = len(v1)

phi_S_degree_weighted_closure_1 = np.sum(
    n_k_closure_1[None, :]
    * k_values_full_closure_1[None, :]
    * phi_S_if_k_closure_1,
    axis=1
) / (2.0 * total_edges_closure_1)

IS_k_closure_1 = np.sum(
    n_k_closure_1[None, :, None]
    * m_grid_closure_1
    * phi_S_m_if_k_closure_1,
    axis=2
)

SI_k_closure_1 = np.sum(
    n_k_closure_1[None, :, None]
    * mbar_grid_closure_1
    * phi_I_m_if_k_closure_1,
    axis=2
)

k_minus_1_grid_closure_1 = np.maximum(
    k_values_full_closure_1[None, :] - 1.0,
    0.0
)

local_product_term_1_closure_1 = np.sum(
    IS_k_closure_1 * phi_S_if_k_closure_1,
    axis=1
)

local_product_term_2_closure_1 = np.sum(
    SI_k_closure_1 * phi_S_if_k_closure_1,
    axis=1
)

local_product_term_3_closure_1 = (
    np.sum(
        IS_k_closure_1 * k_minus_1_grid_closure_1,
        axis=1
    )
    * phi_S_degree_weighted_closure_1
)

local_product_term_4_closure_1 = (
    np.sum(
        SI_k_closure_1 * k_minus_1_grid_closure_1,
        axis=1
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

dt = np.diff(time_grid)

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

# ============================================================
# derivative comparison plot
# ============================================================

plt.figure(figsize=(9, 5.5))

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

plt.axhline(0.0, color="black", linewidth=1, linestyle=":")

plt.xlabel(r"$t$")
plt.ylabel(r"$d\mathrm{Var}(S)/dt$")
plt.title(r"Derivative comparison for $\mathrm{Var}(S)$")
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# integrated variance comparison plot
# ============================================================

plt.figure(figsize=(9, 5.5))

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
    label=r"integrated closure 1",
)

plt.xlabel(r"$t$")
plt.ylabel(r"$\mathrm{Var}(S)$")
plt.title(r"Variance comparison: direct, curve file, full covariance, local covariance")
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# error plot: relative to snapshot variance
# ============================================================

plt.figure(figsize=(9, 5.5))

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

plt.axhline(0.0, color="black", linewidth=1, linestyle=":")

plt.xlabel(r"$t$")
plt.ylabel(r"error")
plt.title(r"Variance errors relative to snapshot $\mathrm{Var}(S)$")
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# error plot: relative to curve-file variance
# ============================================================

plt.figure(figsize=(9, 5.5))

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

plt.axhline(0.0, color="black", linewidth=1, linestyle=":")

plt.xlabel(r"$t$")
plt.ylabel(r"error")
plt.title(r"Variance errors relative to curve-file $\mathrm{Var}(S)$")
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# save all comparison curves
# ============================================================

save_dir = base_dir / "VarianceComparisons"
save_dir.mkdir(parents=True, exist_ok=True)

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
    N_processes=N_processes,
    beta=beta,

    time_grid=time_grid,

    var_S_snapshot_all=var_S_snapshot_all,
    var_S_curve_all=var_S_curve_all,
    var_S_curve_corrected_all=var_S_curve_corrected_all,

    var_S_integrated_from_snapshot_formula=var_S_integrated_from_snapshot_formula,
    var_S_integrated_from_localcov_formula=var_S_integrated_from_localcov_formula,

    # Closure 1 integrated variance
    var_S_integrated_from_closure_1=var_S_integrated_from_closure_1,
    integral_dVarS_dt_closure_1=integral_dVarS_dt_closure_1,
    dVarS_dt_no_triangles_closure_1=dVarS_dt_no_triangles_closure_1,

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
    dVarS_dt_curve_corrected_all=dVarS_dt_curve_corrected_all,
    dVarS_dt_snapshot_all=dVarS_dt_snapshot_all,
    dVarS_dt_localcov_all=dVarS_dt_localcov_all,

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
print("Saved full/local variance comparison curves to:")
print(save_path_compare)