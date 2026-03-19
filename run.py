"""
Define a search window and calculate potential trajectories to Jupiter, minimizing delta-V.
Plot the results in a porkchop plot. For the ideal trajectory, animate spacecraft journey.
"""

import multiprocessing as mp
from datetime import datetime

import numpy as np

import horizons_reader as hr
import lambert_solver as ls
import plotting

# Lambert solution controls
MAX_REVS = 2

# Precompute/pruning control (beam width)
TOPK_DIRECT = 20
TOPK_GA = 80

# Use multiple processes for Lambert precomputation. Tune this for your machine if needed.
NUM_WORKERS = None


def print_solution_summary(label, traj, epochs):
    total_vinf = traj["vinf_launch_kms"] + traj["vinf_arrive_kms"]
    total_tof_days = sum(traj["tof_days"])
    print(f"{label}:")
    print(
        f"  type = {traj['type']}, launch = {traj['vinf_launch_kms']:.3f} km/s, "
        f"arrival = {traj['vinf_arrive_kms']:.3f} km/s, total = {total_vinf:.3f} km/s, "
        f"TOF = {total_tof_days:.1f} days"
    )
    print(f"  sequence = {traj['sequence']}")
    print(f"  dates = {[str(epochs[i]) for i in traj['indices']]}")


def main():
    # define search window and if plot should be annotated with window
    annotate = False
    if annotate:
        start = "2028-11-01"
        stop = "2032-01-01"
        step = 10  # days

        window_info = dict()
        window_info["best_launch"] = datetime(2028, 12, 24)
        window_info["best_arrival"] = datetime(2031, 12, 1)
        window_info["window_start"] = datetime(2028, 12, 13)
        window_info["window_end"] = datetime(2029, 1, 6)
    else:
        window_info = None
        start = "2026-07-01"
        stop = "2055-07-01"
        step = 10  # days

    epochs, bodies = hr.load_bodies(start, stop, f"{step}d")
    elapsed_days = np.arange(0, len(epochs)) * step

    stored, best_by_metric, best_cost = ls.calculate_trajectories(
        bodies, elapsed_days, MAX_REVS, TOPK_DIRECT, TOPK_GA, NUM_WORKERS
    )
    best_total = best_by_metric["total"]

    print("Stored trajectories:", len(stored))
    print(
        f"Best cost (km/s): tot = {best_cost:.3f}, "
        f"launch = {best_total['vinf_launch_kms']:.3f}, "
        f"arrival = {best_total['vinf_arrive_kms']:.3f}"
    )
    print_solution_summary("Best total v_inf trajectory", best_by_metric["total"], epochs)
    print_solution_summary("Best launch v_inf trajectory", best_by_metric["launch"], epochs)
    print_solution_summary(
        "Best arrival v_inf trajectory", best_by_metric["arrival"], epochs
    )

    # create porkchop plots
    plotting.make_plots(stored, epochs, window_info)
    plotting.plot_spacecraft_traj(best_by_metric["total"], epochs, bodies, tag="total")
    plotting.plot_spacecraft_traj(best_by_metric["launch"], epochs, bodies, tag="launch")
    plotting.plot_spacecraft_traj(
        best_by_metric["arrival"], epochs, bodies, tag="arrival"
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
