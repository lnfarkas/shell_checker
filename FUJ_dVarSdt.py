import numpy as np
from pathlib import Path
import time
import matplotlib.pyplot as plt

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

t_start = time.time()
# ============================================================
# find and load graph file
# ============================================================

graph_path, N, v1, v2 = load_graph_file(
    graphs_dir=graphs_dir,
    instance_tag=INSTANCE_TAG,
)

# ==============================================================
# load full simulations and curves
#===============================================================

full_process_dir = base_dir / "FullProcess"
curves_dir = base_dir / "Curves"

N_processes = 1000
approx_time = 0.3

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

snapshot_states_list, snapshot_times, loaded_paths, skipped_paths = load_snapshots_at_or_before_time(
    full_process_paths=full_process_paths,
    matched_curve_time=matched_curve_time,
)

from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# Parallel compute dVar(S)/dt from snapshots for every curve-file time
#
# dVar(S)/dt = beta <[SI]> - 2 beta Cov(S, [SI])
# Cov(S,[SI]) = <S[SI]> - <S><[SI]>
# ============================================================

S_state = 0
I_state = 1

time_grid = curve_data["time_grid"]

dVarS_dt_snapshot_all = np.zeros_like(time_grid, dtype=np.float64)
mean_S_all = np.zeros_like(time_grid, dtype=np.float64)
mean_SI_all = np.zeros_like(time_grid, dtype=np.float64)
mean_S_SI_all = np.zeros_like(time_grid, dtype=np.float64)
cov_S_SI_all = np.zeros_like(time_grid, dtype=np.float64)
var_S_snapshot_all = np.zeros_like(time_grid, dtype=np.float64)
n_loaded_all = np.zeros_like(time_grid, dtype=np.int64)
n_skipped_all = np.zeros_like(time_grid, dtype=np.int64)


def compute_one_time(tidx):
    t = time_grid[tidx]

    snapshot_states_list, snapshot_times, loaded_paths, skipped_paths = load_snapshots_at_or_before_time(
        full_process_paths=full_process_paths,
        matched_curve_time=t,
    )

    S_counts = np.array(
        [
            np.sum(states == S_state)
            for states in snapshot_states_list
        ],
        dtype=np.float64,
    )

    SI_counts = np.array(
        [
            np.sum(
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
            for states in snapshot_states_list
        ],
        dtype=np.float64,
    )

    mean_S = np.mean(S_counts)
    mean_SI = np.mean(SI_counts)
    mean_S_SI = np.mean(S_counts * SI_counts)

    cov_S_SI = mean_S_SI - mean_S * mean_SI

    dVarS_dt_snapshot = (
        beta * mean_SI
        - 2.0 * beta * cov_S_SI
    )

    var_S_snapshot = np.var(S_counts)

    return {
        "tidx": tidx,
        "t": t,
        "mean_S": mean_S,
        "mean_SI": mean_SI,
        "mean_S_SI": mean_S_SI,
        "cov_S_SI": cov_S_SI,
        "dVarS_dt_snapshot": dVarS_dt_snapshot,
        "var_S_snapshot": var_S_snapshot,
        "n_loaded": len(loaded_paths),
        "n_skipped": len(skipped_paths),
    }


n_workers = 15

t_start = time.time()

with ThreadPoolExecutor(max_workers=n_workers) as executor:
    futures = [
        executor.submit(compute_one_time, tidx)
        for tidx in range(len(time_grid))
    ]

    for done_idx, future in enumerate(as_completed(futures), start=1):
        out = future.result()

        tidx = out["tidx"]

        mean_S_all[tidx] = out["mean_S"]
        mean_SI_all[tidx] = out["mean_SI"]
        mean_S_SI_all[tidx] = out["mean_S_SI"]
        cov_S_SI_all[tidx] = out["cov_S_SI"]
        dVarS_dt_snapshot_all[tidx] = out["dVarS_dt_snapshot"]
        var_S_snapshot_all[tidx] = out["var_S_snapshot"]
        n_loaded_all[tidx] = out["n_loaded"]
        n_skipped_all[tidx] = out["n_skipped"]

        # print(
        #     f"Finished {done_idx}/{len(time_grid)} "
        #     f"(tidx={tidx}, t={out['t']}, loaded={out['n_loaded']}, skipped={out['n_skipped']})"
        # )

t_end = time.time()

print()
print("Finished computing dVar(S)/dt from snapshots for all times.")
print("Total time =", t_end - t_start, "seconds")

# ============================================================
# Compare snapshot formula with numerical derivative from curve file
# ============================================================

var_S_curve_all = curve_data["var_fractions"][:, 0] * N**2

dVarS_dt_curve_all = np.gradient(
    var_S_curve_all,
    time_grid
)

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
# Integral of dVar(S)/dt from snapshot formula
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

# Use initial snapshot variance as initial condition
var_S_integrated_from_snapshot_formula = (
    var_S_snapshot_all[0]
    + integral_dVarS_dt_snapshot
)


# ============================================================
# Save Var(S) from integrated covariance formula
# This is the exact curve plotted as:
# var_S_integrated_from_snapshot_formula
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

    # the actual plotted integrated covariance-formula curve
    var_S_integrated_from_snapshot_formula=var_S_integrated_from_snapshot_formula,

    # ingredients used to construct it
    dVarS_dt_snapshot_all=dVarS_dt_snapshot_all,
    cov_S_SI_all=cov_S_SI_all,
    mean_SI_all=mean_SI_all,
    mean_S_all=mean_S_all,
    mean_S_SI_all=mean_S_SI_all,

    # initial condition used
    var_S_initial=var_S_snapshot_all[0],

    # optional comparison curves
    var_S_snapshot_all=var_S_snapshot_all,
    var_S_curve_all=var_S_curve_all,
)

print()
print("Saved integrated covariance-formula Var(S) curve to:")
print(save_path)

# Curve-file variance
var_S_curve_all = curve_data["var_fractions"][:, 0] * N**2

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
    label=r"$\mathrm{Var}(S)(0)+\int_0^t d\mathrm{Var}(S)/dt\,du$ from snapshot formula",
)

plt.plot(
    time_grid,
    var_S_curve_all,
    linestyle=":",
    linewidth=2,
    label=r"$\mathrm{Var}(S)$ from curve file",
)

plt.xlabel(r"$t$")
plt.ylabel(r"$\mathrm{Var}(S)$")
plt.title(r"Integrated snapshot formula for $\mathrm{Var}(S)(t)$")
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

t_end = time.time()

print()
print("Total execution time:", (t_end - t_start) / 60, "minutes")