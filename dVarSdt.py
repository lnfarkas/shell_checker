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

n_workers = 8

t_total_start = time.time()

# ============================================================
# load graph
# ============================================================

graph_path, N, v1, v2 = load_graph_file(
    graphs_dir=graphs_dir,
    instance_tag=INSTANCE_TAG,
)

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
# plot derivative comparison
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    time_grid,
    dVarS_dt_snapshot_all,
    linewidth=2,
    label=r"$\beta\langle[SI]\rangle - 2\beta\,\mathrm{Cov}(S,[SI])$ from snapshots",
)

plt.plot(
    time_grid,
    dVarS_dt_curve_all,
    linestyle="--",
    linewidth=2,
    label=r"numerical derivative of curve-file $\mathrm{Var}(S)$",
)

plt.axhline(0.0, color="black", linewidth=1, linestyle=":")

plt.xlabel(r"$t$")
plt.ylabel(r"$d\mathrm{Var}(S)/dt$")
plt.title(r"$d\mathrm{Var}(S)/dt$: snapshot formula vs curve-file derivative")
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

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
# plot integrated variance
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    time_grid,
    var_S_snapshot_all,
    linewidth=2,
    label=r"$\mathrm{Var}(S)$ from snapshots",
)

plt.plot(
    time_grid,
    var_S_integrated_from_snapshot_formula,
    linestyle="--",
    linewidth=2,
    label=r"$\mathrm{Var}(S)(0)+\int_0^t d\mathrm{Var}(S)/dt\,du$ from covariance formula",
)

plt.plot(
    time_grid,
    var_S_curve_all*(N-1)/N,
    linestyle=":",
    linewidth=2,
    label=r"$\mathrm{Var}(S)$ from curve file",
)

plt.xlabel(r"$t$")
plt.ylabel(r"$\mathrm{Var}(S)$")
plt.title(r"Integrated covariance formula for $\mathrm{Var}(S)(t)$")
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# error plot
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    time_grid,
    var_S_integrated_from_snapshot_formula - var_S_snapshot_all,
    linewidth=2,
    label=r"integrated covariance formula $-$ snapshot variance",
)

plt.plot(
    time_grid,
    var_S_integrated_from_snapshot_formula - var_S_curve_all*(N-1)/N,
    linestyle="--",
    linewidth=2,
    label=r"integrated covariance formula $-$ curve-file variance",
)

plt.plot(
    time_grid,
    var_S_snapshot_all - var_S_curve_all*(N-1)/N,
    linestyle="--",
    linewidth=2,
    label=r"snapshot variance $-$ curve-file variance",
)

plt.axhline(0.0, color="black", linewidth=1, linestyle=":")

plt.xlabel(r"$t$")
plt.ylabel(r"error")
plt.title(r"Error of integrated covariance formula")
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# total time
# ============================================================

t_total_end = time.time()

print()
print("Total execution time:", t_total_end - t_total_start, "seconds")
print("Total execution time:", (t_total_end - t_total_start) / 60, "minutes")