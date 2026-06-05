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

shell_n = 2


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

summary = summarize_covariance_parallel_streaming(
    full_process_paths=full_process_paths,
    matched_curve_time=matched_curve_time,
    N=N,
    v1=v1,
    v2=v2,
    shell_n_nodes=shell_n_nodes,
    S_state=0,
    I_state=1,
    n_workers=8,
    chunk_size=20,
)

print_covariance_summary(summary)

summary = summarize_covariance_parallel_streaming(
    full_process_paths=full_process_paths,
    matched_curve_time=matched_curve_time,
    N=N,
    v1=v1,
    v2=v2,
    shell_n_nodes=shell_n_nodes,
    S_state=0,
    I_state=1,
    n_workers=8,
    chunk_size=20,
)

print_covariance_summary(summary)

# ============================================================
# shell analysis
# ============================================================

shell_sizes = np.asarray(
    [x.size for x in shell_n_nodes],
    dtype=np.int64,
)

print()
print("Shell size diagnostics")
print("----------------------")
print("shell_n =", shell_n)
print("number of centers with nonempty shell =", np.sum(shell_sizes > 0))
print("fraction of centers with nonempty shell =", np.mean(shell_sizes > 0))
print("mean shell size =", np.mean(shell_sizes))
print("std shell size  =", np.std(shell_sizes))
print("min shell size  =", np.min(shell_sizes))
print("max shell size  =", np.max(shell_sizes))
print("total shell nodes counted over centers =", np.sum(shell_sizes))

if np.sum(shell_sizes > 0) > 0:
    print("mean shell size among nonempty shells =", np.mean(shell_sizes[shell_sizes > 0]))
else:
    print("mean shell size among nonempty shells = nan")


# ============================================================
# shell-edge analysis
# ============================================================

offsets = summary["offsets"]

shell_edge_sizes = offsets[1:] - offsets[:-1]

print()
print("Shell-edge diagnostics")
print("----------------------")
print("number of centers with nonempty shell-edge set =", np.sum(shell_edge_sizes > 0))
print("fraction of centers with nonempty shell-edge set =", np.mean(shell_edge_sizes > 0))
print("mean shell-edge count =", np.mean(shell_edge_sizes))
print("std shell-edge count  =", np.std(shell_edge_sizes))
print("min shell-edge count  =", np.min(shell_edge_sizes))
print("max shell-edge count  =", np.max(shell_edge_sizes))
print("total shell edges counted over centers =", np.sum(shell_edge_sizes))

if np.sum(shell_edge_sizes > 0) > 0:
    print(
        "mean shell-edge count among nonempty shell-edge sets =",
        np.mean(shell_edge_sizes[shell_edge_sizes > 0]),
    )
else:
    print("mean shell-edge count among nonempty shell-edge sets = nan")


# ============================================================
# edge density around shell
# ============================================================

shell_edge_per_shell_node = np.full(
    shell_sizes.shape,
    np.nan,
    dtype=np.float64,
)

nonempty_shell = shell_sizes > 0

shell_edge_per_shell_node[nonempty_shell] = (
    shell_edge_sizes[nonempty_shell]
    /
    shell_sizes[nonempty_shell]
)

print()
print("Shell-edge per shell-node diagnostics")
print("-------------------------------------")
print("mean shell-edge / shell-node =", np.nanmean(shell_edge_per_shell_node))
print("std shell-edge / shell-node  =", np.nanstd(shell_edge_per_shell_node))
print("min shell-edge / shell-node  =", np.nanmin(shell_edge_per_shell_node))
print("max shell-edge / shell-node  =", np.nanmax(shell_edge_per_shell_node))