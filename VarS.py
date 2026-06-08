import numpy as np
from pathlib import Path
import time

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

# ============================================================
# Compute <S^2> - <S>^2 from full-process snapshots
# ============================================================

S_counts = np.array(
    [
        np.sum(states == 0)   # S_state = 0
        for states in snapshot_states_list
    ],
    dtype=np.float64,
)

mean_S = np.mean(S_counts)
mean_S2 = np.mean(S_counts**2)

var_S_from_snapshots = mean_S2 - mean_S**2

print()
print("Variance of S from snapshots")
print("----------------------------")
print("number of realizations =", len(S_counts))
print("<S>     =", mean_S)
print("<S^2>   =", mean_S2)
print("<S^2> - <S>^2 =", var_S_from_snapshots)

# ============================================================
# Compare:
# 1. <S^2> - <S>^2 from snapshots
# 2. np.var(S_counts) from snapshots
# 3. Var(S) from curve file at matched time
# ============================================================

var_S_np_var_population = np.var(S_counts)          # same as <S^2> - <S>^2
var_S_np_var_unbiased = np.var(S_counts, ddof=1)    # sample/unbiased version

mean_S_from_curve = curve_data["mean_fractions"][curve_idx, 0] * N
var_S_from_curve = curve_data["var_fractions"][curve_idx, 0] * N**2

print()
print("Compare Var(S) estimates")
print("------------------------")

print()
print("Means")
print("-----")
print("<S> from snapshots =", mean_S)
print("<S> from curve     =", mean_S_from_curve)
print("difference         =", mean_S - mean_S_from_curve)

print()
print("Variances")
print("---------")
print("<S^2> - <S>^2 from snapshots =", var_S_from_snapshots)
print("np.var(S_counts) population  =", var_S_np_var_population)
print("np.var(S_counts, ddof=1)     =", var_S_np_var_unbiased)
print("Var(S) from curve            =", var_S_from_curve)

print()
print("Differences from curve")
print("----------------------")
print("mean_S2 - mean_S**2 from snapshots - curve variance =", var_S_from_snapshots - var_S_from_curve)
print("np.var population - curve variance        =", var_S_np_var_population - var_S_from_curve)
print("np.var ddof=1 - curve variance            =", var_S_np_var_unbiased - var_S_from_curve)

print()
print("Relative differences from curve")
print("--------------------------------")
print(
    "mean_S2 - mean_S**2 from snapshots:",
    (var_S_from_snapshots - var_S_from_curve) / var_S_from_curve
)
print(
    "np.var population:",
    (var_S_np_var_population - var_S_from_curve) / var_S_from_curve
)
print(
    "np.var ddof=1:",
    (var_S_np_var_unbiased - var_S_from_curve) / var_S_from_curve
)

# ============================================================
# Compute <S> from full-process snapshots
# ============================================================

S_state = 0

S_counts = np.array(
    [
        np.sum(states == S_state)
        for states in snapshot_states_list
    ],
    dtype=np.float64,
)

mean_S = np.mean(S_counts)

print()
print("Mean S from snapshots")
print("---------------------")
print("<S> =", mean_S)

# ============================================================
# Compute <[SI]> from full-process snapshots
# [SI] = number of S-I edges, counted once
# ============================================================

S_state = 0
I_state = 1

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

mean_SI = np.mean(SI_counts)

print()
print("Mean [SI] from snapshots")
print("------------------------")
print("<[SI]> =", mean_SI)

# ============================================================
# Compute <S[SI]> from full-process snapshots
# ============================================================

S_state = 0
I_state = 1

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

S_times_SI_counts = S_counts * SI_counts

mean_S_times_SI = np.mean(S_times_SI_counts)

factorized_S_SI = np.mean(S_counts) * np.mean(SI_counts)

cov_S_SI = mean_S_times_SI - factorized_S_SI

print()
print("Mean S[SI] from snapshots")
print("-------------------------")
print("<S[SI]>       =", mean_S_times_SI)
print("<S><[SI]>     =", factorized_S_SI)
print("<S[SI]> - <S><[SI]> =", cov_S_SI)

# ============================================================
# Calculate dVar(S)/dt = beta <[SI]> - 2 beta Cov(S, [SI])
# where Cov(S, [SI]) = <S[SI]> - <S><[SI]>
# ============================================================

cov_S_SI = mean_S_times_SI - factorized_S_SI

dVarS_dt_from_snapshots = (
    beta * mean_SI
    - 2.0 * beta * cov_S_SI
)

print()
print("dVar(S)/dt from snapshots")
print("-------------------------")
print("beta =", beta)
print("dVar(S)/dt = beta <[SI]> - 2 beta Cov(S,[SI]) =", dVarS_dt_from_snapshots)