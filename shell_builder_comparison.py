import numpy as np
from pathlib import Path
import time

from shell_builders import *

# ============================================================
# helper functions for comparison
# ============================================================

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

graph_paths = sorted(graphs_dir.glob(f"graph_*_{INSTANCE_TAG}_*.npz"))

if len(graph_paths) == 0:
    raise FileNotFoundError(
        f"No graph file found for {INSTANCE_TAG} in {graphs_dir}"
    )

if len(graph_paths) > 1:
    raise RuntimeError(
        f"More than one graph file found for {INSTANCE_TAG}:\n"
        + "\n".join(str(p) for p in graph_paths)
    )

graph_path = graph_paths[0]

print(f"Using graph:\n{graph_path}")

with np.load(graph_path, allow_pickle=False) as graph_data:
    N = int(graph_data["N_vertices_in_LCC"])
    v1 = graph_data["v1_sorted"].astype(np.int64, copy=False)
    v2 = graph_data["v2_sorted_by_v1"].astype(np.int64, copy=False)

print(f"N = {N}")
print(f"E = {len(v1)}")


# ============================================================
# benchmark
# ============================================================

print("\n" + "=" * 80)
print(f"Benchmarking shell_n = {shell_n}")
print("=" * 80)

# -------------------------------
# set/frontier version
# -------------------------------

t0 = time.time()

neighbors = build_neighbors_from_edges(N, v1, v2)

t1 = time.time()

shell_set = build_shell_n_nodes_set(
    N=N,
    neighbors=neighbors,
    shell_n=shell_n,
    dtype=np.int32,
)

t2 = time.time()

# -------------------------------
# dense NumPy version
# -------------------------------

shell_dense = build_shell_n_nodes_dense_numpy(
    N=N,
    v1=v1,
    v2=v2,
    shell_n=shell_n,
    dtype=np.int32,
)

t3 = time.time()

# -------------------------------
# correctness and timings
# -------------------------------

same = shell_lists_equal(shell_set, shell_dense)

print("\nCorrectness")
print("-----------")
print("same shells:", same)

print("\nTiming")
print("------")
print(f"neighbor build time              = {t1 - t0:.6f} seconds")
print(f"set/frontier shell time only     = {t2 - t1:.6f} seconds")
print(f"set/frontier total time          = {t2 - t0:.6f} seconds")
print(f"dense NumPy matrix shell time    = {t3 - t2:.6f} seconds")

if (t3 - t2) > 0:
    print(f"set shell / dense ratio          = {(t2 - t1) / (t3 - t2):.6f}")
    print(f"set total / dense ratio          = {(t2 - t0) / (t3 - t2):.6f}")

print("\nShell sizes")
print("-----------")
shell_size_summary(shell_set)