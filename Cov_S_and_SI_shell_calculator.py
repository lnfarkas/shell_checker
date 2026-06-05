import numpy as np
from pathlib import Path
import time

from loaders import *
from X_and_SI_shell_builders import *

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

shell_n = 6


# ============================================================
# find and load graph file
# ============================================================

graph_path, N, v1, v2 = load_graph_file(
    graphs_dir=graphs_dir,
    instance_tag=INSTANCE_TAG,
)

# ============================================================
# build shell-n nodes
# ============================================================

t1 = time.time()

neighbors = build_neighbors_from_edges(N, v1, v2)

shell_n_nodes = build_shell_n_nodes_set(
    N=N,
    neighbors=neighbors,
    shell_n=shell_n,
    dtype=np.int32,
)

t2 = time.time()

print(
    f"Time to compute shell-{shell_n} nodes for all N = {N} nodes:",
    t2 - t1,
    "seconds",
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
# build X_i^(r) and ([SI]_{nth_i})^(r) matrices
# ============================================================

S_state = 0
I_state = 1

t3 = time.time()

# Precompute, for every center node i, which graph edges touch shell_n_nodes[i]
shell_edge_indices = build_shell_edge_indices(
    N=N,
    v1=v1,
    v2=v2,
    shell_n_nodes=shell_n_nodes,
    dtype=np.int32,
)

N_loaded = len(snapshot_states_list)

X_matrix = np.zeros((N_loaded, N), dtype=np.int8)
SI_shell_matrix = np.zeros((N_loaded, N), dtype=np.int64)

for r, snapshot_states in enumerate(snapshot_states_list):
    X_row, SI_shell_row = compute_X_and_SI_shell_for_snapshot(
        snapshot_states=snapshot_states,
        v1=v1,
        v2=v2,
        shell_edge_indices=shell_edge_indices,
        S_state=S_state,
        I_state=I_state,
        dtype_X=np.int8,
        dtype_SI=np.int64,
    )

    X_matrix[r, :] = X_row
    SI_shell_matrix[r, :] = SI_shell_row

t4 = time.time()

print("\nBuilt matrices")
print("--------------")
print("X_matrix.shape        =", X_matrix.shape)
print("SI_shell_matrix.shape =", SI_shell_matrix.shape)
print("matrix build time     =", t4 - t3, "seconds")

# ============================================================
# covariance summary
# ============================================================

summary = summarize_covariance_from_X_and_SI_shell(
    X_matrix=X_matrix,
    SI_shell_matrix=SI_shell_matrix,
)

print("\nCovariance summary")
print("------------------")
print("mean_S_SI_nth       =", summary["mean_S_SI_nth"])
print("factorized          =", summary["factorized"])
print("covariance          =", summary["covariance"])
print("relative_covariance =", summary["relative_covariance"])
