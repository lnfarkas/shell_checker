import numpy as np
from pathlib import Path

# ==================================================================================
# graph
# ==================================================================================

def load_graph_file(graphs_dir, instance_tag):
    """
    Finds and loads the unique graph file for one instance.

    Parameters
    ----------
    graphs_dir : str or Path
        Path to the Graphs directory.

    instance_tag : str
        Example: "instanceNo0000"

    Returns
    -------
    graph_path : Path
        Path to the selected graph file.

    N : int
        Number of vertices in the LCC.

    v1 : ndarray
        First endpoint array.

    v2 : ndarray
        Second endpoint array.
    """

    graphs_dir = Path(graphs_dir)

    graph_paths = sorted(graphs_dir.glob(f"graph_*_{instance_tag}_*.npz"))

    if len(graph_paths) == 0:
        raise FileNotFoundError(
            f"No graph file found for {instance_tag} in {graphs_dir}"
        )

    if len(graph_paths) > 1:
        raise RuntimeError(
            f"More than one graph file found for {instance_tag}:\n"
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

    return graph_path, N, v1, v2

# ==================================================================================
# full simulation
# ==================================================================================

def find_full_process_paths(
    full_process_dir,
    instance_tag,
    n_processes=None,
):
    """
    Finds full_process files for one graph instance.

    Parameters
    ----------
    full_process_dir : str or Path
        Path to the FullProcess directory.

    instance_tag : str
        Example: "instanceNo0000"

    n_processes : int or None
        If int, keep only the first n_processes files.
        If None, keep all matching files.

    Returns
    -------
    full_process_paths : list[Path]
    """

    full_process_dir = Path(full_process_dir)

    all_full_process_paths = sorted(full_process_dir.glob("full_process_*.npz"))

    print(f"Found {len(all_full_process_paths)} total full_process files.")

    full_process_paths_all_instance = [
        p for p in all_full_process_paths
        if instance_tag in p.name
    ]

    if len(full_process_paths_all_instance) == 0:
        print("\nFirst few files actually found in FullProcess:")
        for p in all_full_process_paths[:20]:
            print(" ", p.name)

        raise FileNotFoundError(
            f"No full_process files found containing {instance_tag} in {full_process_dir}"
        )

    if n_processes is not None:
        if len(full_process_paths_all_instance) < n_processes:
            raise ValueError(
                f"Wanted n_processes = {n_processes}, "
                f"but only found {len(full_process_paths_all_instance)} files "
                f"for {instance_tag}"
            )

        full_process_paths = full_process_paths_all_instance[:n_processes]
    else:
        full_process_paths = full_process_paths_all_instance

    print(
        f"Using {len(full_process_paths)} full_process files for {instance_tag}."
    )

    return full_process_paths


def load_snapshot_at_or_before_time(full_process_path, matched_curve_time):
    """
    Loads one full_process file and selects the last raw snapshot with

        raw_time <= matched_curve_time

    Parameters
    ----------
    full_process_path : str or Path
        Path to one full_process_*.npz file.

    matched_curve_time : float
        Time at which we want the latest available raw snapshot.

    Returns
    -------
    loaded : bool
        False if matched_curve_time is before the first raw time.

    snapshot_time : float or None

    snapshot_states : ndarray or None
    """

    full_process_path = Path(full_process_path)

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


def load_snapshots_at_or_before_time(full_process_paths, matched_curve_time):
    """
    Loads snapshots from many full_process files.

    Returns only processes where at least one raw_time <= matched_curve_time.

    Returns
    -------
    snapshot_states_list : list[ndarray]

    snapshot_times : ndarray

    loaded_paths : list[Path]

    skipped_paths : list[Path]
    """

    snapshot_states_list = []
    snapshot_times = []
    loaded_paths = []
    skipped_paths = []

    for full_process_path in full_process_paths:
        loaded, snapshot_time, snapshot_states = load_snapshot_at_or_before_time(
            full_process_path=full_process_path,
            matched_curve_time=matched_curve_time,
        )

        if not loaded:
            skipped_paths.append(Path(full_process_path))
            continue

        snapshot_states_list.append(snapshot_states)
        snapshot_times.append(snapshot_time)
        loaded_paths.append(Path(full_process_path))

    if len(snapshot_states_list) == 0:
        raise RuntimeError(
            "No snapshots were loaded. matched_curve_time is before the first raw time "
            "in all processes."
        )

    snapshot_times = np.asarray(snapshot_times, dtype=np.float64)

    print("\nLoaded snapshots")
    print("----------------")
    print(f"loaded  = {len(loaded_paths)}")
    print(f"skipped = {len(skipped_paths)}")

    return snapshot_states_list, snapshot_times, loaded_paths, skipped_paths

# ==================================================================================
# curves
# ==================================================================================

def find_curve_path(
    curves_dir,
    instance_tag,
    n_processes,
):
    """
    Finds the unique curve file matching instance_tag and Nprocesses tag.

    Parameters
    ----------
    curves_dir : str or Path
        Path to Curves directory.

    instance_tag : str
        Example: "instanceNo0000"

    n_processes : int
        Example: 1000

    Returns
    -------
    curve_path : Path
    """

    curves_dir = Path(curves_dir)
    n_process_tag = f"Nprocesses{n_processes}"

    curve_paths = sorted(
        p for p in curves_dir.glob("curves_*.npz")
        if instance_tag in p.name
        and n_process_tag in p.name
    )

    if len(curve_paths) == 0:
        print("\nCurve files actually found:")
        for p in sorted(curves_dir.glob("curves_*.npz")):
            print(" ", p.name)

        raise FileNotFoundError(
            f"No curve file found containing both {instance_tag} "
            f"and {n_process_tag} in {curves_dir}"
        )

    if len(curve_paths) > 1:
        raise RuntimeError(
            f"More than one curve file found for {instance_tag}, {n_process_tag}:\n"
            + "\n".join(str(p) for p in curve_paths)
        )

    curve_path = curve_paths[0]

    print(f"Using curve file:\n{curve_path}")

    return curve_path


def load_curve_data(curve_path):
    """
    Loads the basic arrays from one curves_*.npz file.

    Returns
    -------
    curve_data_dict : dict
    """

    curve_path = Path(curve_path)

    with np.load(curve_path, allow_pickle=False) as curve_data:
        out = {
            key: curve_data[key].copy()
            for key in curve_data.files
        }

    return out


def match_approx_time_to_curve_time(curve_path, approx_time):
    """
    Finds the curve time closest to approx_time.

    Parameters
    ----------
    curve_path : str or Path

    approx_time : float

    Returns
    -------
    curve_idx : int

    matched_curve_time : float

    curve_data_dict : dict
        All arrays from the curve file.
    """

    curve_data_dict = load_curve_data(curve_path)

    curve_time_grid = curve_data_dict["time_grid"]

    curve_idx = int(np.argmin(np.abs(curve_time_grid - approx_time)))
    matched_curve_time = float(curve_time_grid[curve_idx])

    print("\nMatched curve time")
    print("------------------")
    print(f"approx_time        = {approx_time}")
    print(f"curve_idx          = {curve_idx}")
    print(f"matched_curve_time = {matched_curve_time}")

    return curve_idx, matched_curve_time, curve_data_dict


def load_curve_and_match_time(
    curves_dir,
    instance_tag,
    n_processes,
    approx_time,
):
    """
    Convenience wrapper:

    1. find curve file
    2. load curve file
    3. match approx_time to closest curve time

    Returns
    -------
    curve_path : Path

    curve_idx : int

    matched_curve_time : float

    curve_data_dict : dict
    """

    curve_path = find_curve_path(
        curves_dir=curves_dir,
        instance_tag=instance_tag,
        n_processes=n_processes,
    )

    curve_idx, matched_curve_time, curve_data_dict = match_approx_time_to_curve_time(
        curve_path=curve_path,
        approx_time=approx_time,
    )

    return curve_path, curve_idx, matched_curve_time, curve_data_dict