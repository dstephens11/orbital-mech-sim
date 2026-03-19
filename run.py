"""
Define a search window and calculate potential trajectories to Jupiter, minimizing delta-V.
Plot the results in a porkchop plot. For the ideal trajectory, animate spacecraft journey.
"""

import argparse
import json
import multiprocessing as mp
from datetime import datetime
from pathlib import Path

import numpy as np

import horizons_reader as hr
import lambert_solver as ls
import plotting
import constants as c

# Lambert solution controls
MAX_REVS = 2

# Precompute/pruning control (beam width)
TOPK_DIRECT = 20
TOPK_GA = 80

# Use multiple processes for Lambert precomputation. Tune this for your machine if needed.
NUM_WORKERS = None

RESULTS_ROOT = Path("results")


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
    print(f"  dates = {[str(epochs[i]) for i in traj['indices']]}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Search Earth-to-Jupiter direct and gravity-assist trajectories."
    )
    parser.add_argument(
        "--start", default="2026-07-01", help="Search window start date (YYYY-MM-DD)."
    )
    parser.add_argument(
        "--stop", default="2055-07-01", help="Search window stop date (YYYY-MM-DD)."
    )
    parser.add_argument(
        "--step", type=int, default=10, help="Ephemeris cadence in days."
    )
    parser.add_argument(
        "--max-years",
        type=float,
        default=10.0,
        help="Maximum total trajectory duration in years.",
    )
    parser.add_argument(
        "--annotate", action="store_true", help="Generate annotated porkchop plots."
    )
    parser.add_argument(
        "--max-revs", type=int, default=MAX_REVS, help="Maximum Lambert revolutions."
    )
    parser.add_argument(
        "--topk-direct",
        type=int,
        default=TOPK_DIRECT,
        help="Number of direct Earth-to-Jupiter Lambert legs kept per departure epoch.",
    )
    parser.add_argument(
        "--topk-ga",
        type=int,
        default=TOPK_GA,
        help="Number of gravity-assist Lambert legs kept per departure epoch.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=NUM_WORKERS,
        help="Multiprocessing worker count for Lambert precomputation. Default uses auto selection.",
    )
    parser.add_argument(
        "--thresh-kms",
        type=float,
        default=ls.THRESH_KMS,
        help="Store trajectories with launch or arrival v_inf below this km/s threshold.",
    )
    parser.add_argument(
        "--h-min-km", type=float, default=c.H_MIN, help="Minimum flyby altitude in km."
    )
    parser.add_argument(
        "--h-max-km", type=float, default=c.H_MAX, help="Maximum flyby altitude in km."
    )
    parser.add_argument(
        "--vinf-match-abs-kms",
        type=float,
        default=c.VINF_MATCH_ABS_KMS,
        help="Maximum allowed incoming/outgoing v_inf magnitude mismatch for unpowered flybys in km/s.",
    )
    parser.add_argument(
        "--window-best-launch",
        default="2028-12-24",
        help="Annotated porkchop reference launch date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--window-best-arrival",
        default="2031-12-01",
        help="Annotated porkchop reference arrival date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--window-start",
        default="2028-12-13",
        help="Annotated porkchop window start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--window-end",
        default="2029-01-06",
        help="Annotated porkchop window end date (YYYY-MM-DD).",
    )
    return parser.parse_args()


def build_run_config(args):
    return {
        "start": args.start,
        "stop": args.stop,
        "step_days": args.step,
        "max_years": args.max_years,
        "annotate": args.annotate,
        "max_revs": args.max_revs,
        "topk_direct": args.topk_direct,
        "topk_ga": args.topk_ga,
        "num_workers": args.num_workers,
        "thresh_kms": ls.THRESH_KMS,
        "h_min_km": c.H_MIN,
        "h_max_km": c.H_MAX,
        "vinf_match_abs_kms": c.VINF_MATCH_ABS_KMS,
    }


def make_run_output_dir(config):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = RESULTS_ROOT / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def serialize_traj(traj, epochs):
    return {
        "type": traj["type"],
        "sequence": traj["sequence"],
        "indices": traj["indices"],
        "dates": [str(epochs[i]) for i in traj["indices"]],
        "tof_days": traj["tof_days"],
        "vinf_launch_kms": traj["vinf_launch_kms"],
        "vinf_arrive_kms": traj["vinf_arrive_kms"],
        "vinf_total_kms": traj["vinf_launch_kms"] + traj["vinf_arrive_kms"],
    }


def main():
    args = parse_args()

    ls.THRESH_KMS = args.thresh_kms
    c.H_MIN = args.h_min_km
    c.H_MAX = args.h_max_km
    c.VINF_MATCH_ABS_KMS = args.vinf_match_abs_kms

    if args.annotate:
        window_info = {
            "best_launch": datetime.strptime(args.window_best_launch, "%Y-%m-%d"),
            "best_arrival": datetime.strptime(args.window_best_arrival, "%Y-%m-%d"),
            "window_start": datetime.strptime(args.window_start, "%Y-%m-%d"),
            "window_end": datetime.strptime(args.window_end, "%Y-%m-%d"),
        }
    else:
        window_info = None

    run_config = build_run_config(args)
    output_dir = make_run_output_dir(run_config)
    print(f"Writing results to: {output_dir}")

    epochs, bodies = hr.load_bodies(args.start, args.stop, f"{args.step}d")
    elapsed_days = np.arange(0, len(epochs)) * args.step

    stored, best_by_metric, best_cost = ls.calculate_trajectories(
        bodies,
        elapsed_days,
        args.max_revs,
        args.topk_direct,
        args.topk_ga,
        args.num_workers,
        args.max_years,
    )

    print("\n\n")
    print_solution_summary(
        "Best total v_inf trajectory", best_by_metric["total"], epochs
    )
    print_solution_summary(
        "Best launch v_inf trajectory", best_by_metric["launch"], epochs
    )
    print_solution_summary(
        "Best arrival v_inf trajectory", best_by_metric["arrival"], epochs
    )

    summary = {
        "config": run_config,
        "stored_trajectories": len(stored),
        "best_total": serialize_traj(best_by_metric["total"], epochs),
        "best_launch": serialize_traj(best_by_metric["launch"], epochs),
        "best_arrival": serialize_traj(best_by_metric["arrival"], epochs),
    }
    (output_dir / "run_config.json").write_text(json.dumps(summary, indent=2))

    # create porkchop plots
    plotting.make_plots(stored, epochs, window_info, output_dir=output_dir)
    plotting.plot_spacecraft_traj(
        best_by_metric["total"], epochs, bodies, tag="total", output_dir=output_dir
    )
    plotting.plot_spacecraft_traj(
        best_by_metric["launch"], epochs, bodies, tag="launch", output_dir=output_dir
    )
    plotting.plot_spacecraft_traj(
        best_by_metric["arrival"], epochs, bodies, tag="arrival", output_dir=output_dir
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
