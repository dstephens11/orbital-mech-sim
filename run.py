"""Single command-line entry point for the trajectory search workflow."""

import argparse
import json
import multiprocessing as mp
from datetime import datetime
from pathlib import Path

from arrival import jupiter_capture as jc
import constants as c
import snapshot_io
from search import lambert as ls
from search import refinement
from visualization import plots

MAX_REVS = 2
TOPK_DIRECT = 20
TOPK_GA = 80
NUM_WORKERS = None
RESULTS_ROOT = Path("results")


def print_solution_summary(label, traj, epochs):
    """Print a concise console summary for one best trajectory."""
    total_vinf = traj["vinf_launch_kms"] + traj["vinf_arrive_kms"]
    total_tof_days = sum(traj["tof_days"])
    joi_delta_v_kms = traj.get("joi_delta_v_kms")
    mission_total_cost_kms = traj.get("mission_total_cost_kms")
    print(f"{label}:")
    print(
        f"  type = {traj['type']}, launch = {traj['vinf_launch_kms']:.3f} km/s, "
        f"arrival = {traj['vinf_arrive_kms']:.3f} km/s, total = {total_vinf:.3f} km/s, "
        f"TOF = {total_tof_days:.1f} days"
    )
    if joi_delta_v_kms is not None and mission_total_cost_kms is not None:
        print(
            f"  JOI = {joi_delta_v_kms:.3f} km/s, "
            f"launch + JOI = {mission_total_cost_kms:.3f} km/s"
        )
    print(f"  sequence = {traj['sequence']}")
    print(f"  dates = {[str(epochs[i]) for i in traj['indices']]}\n")


def parse_args(argv=None):
    """Parse command-line configuration for the adaptive search workflow."""
    parser = argparse.ArgumentParser(
        description="Search Earth-to-Jupiter direct and gravity-assist trajectories."
    )
    parser.add_argument("--start", default="2026-07-01", help="Search window start date (YYYY-MM-DD).")
    parser.add_argument("--stop", default="2055-07-01", help="Search window stop date (YYYY-MM-DD).")
    parser.add_argument("--step", type=int, default=10, help="Coarse global ephemeris cadence in days.")
    parser.add_argument("--max-years", type=float, default=10.0, help="Maximum total trajectory duration in years.")
    parser.add_argument("--annotate", action="store_true", help="Generate annotated porkchop plots.")
    parser.add_argument(
        "--refine-steps",
        default=None,
        help="Comma-separated refinement cadences in days. Defaults to halving the coarse cadence down to 1 day.",
    )
    parser.add_argument(
        "--refine-top-n",
        type=int,
        default=3,
        help="Number of best corridors to carry into each refinement level.",
    )
    parser.add_argument(
        "--refine-pad-scale",
        type=float,
        default=12.0,
        help="Half-width of each refined corridor in units of the next cadence.",
    )
    parser.add_argument("--max-revs", type=int, default=MAX_REVS, help="Maximum Lambert revolutions.")
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
        help="Number of gravity-assist-related Lambert legs kept per departure epoch.",
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
    parser.add_argument("--h-min-km", type=float, default=c.H_MIN, help="Minimum flyby altitude in km.")
    parser.add_argument("--h-max-km", type=float, default=c.H_MAX, help="Maximum flyby altitude in km.")
    parser.add_argument(
        "--vinf-match-abs-kms",
        type=float,
        default=c.VINF_MATCH_ABS_KMS,
        help="Maximum allowed incoming/outgoing v_inf magnitude mismatch for unpowered flybys in km/s.",
    )
    parser.add_argument(
        "--joi-periapsis-rj",
        type=float,
        default=c.JOI_CAPTURE_PERIAPSIS_RJ,
        help="Target Jupiter capture-orbit periapsis radius in Jupiter radii.",
    )
    parser.add_argument(
        "--joi-apoapsis-rj",
        type=float,
        default=c.JOI_CAPTURE_APOAPSIS_RJ,
        help="Target Jupiter capture-orbit apoapsis radius in Jupiter radii.",
    )

    args = parser.parse_args(argv)
    refine_steps = refinement.parse_csv_ints(args.refine_steps)
    if refine_steps is None:
        refine_steps = refinement.default_refine_steps(args.step)
    args.refine_steps = [step for step in refine_steps if step < args.step]
    return args


def build_run_config(args):
    """Serialize the core CLI settings used for one run."""
    return {
        "start": args.start,
        "stop": args.stop,
        "step_days": args.step,
        "refine_steps": args.refine_steps,
        "refine_top_n": args.refine_top_n,
        "refine_pad_scale": args.refine_pad_scale,
        "max_years": args.max_years,
        "annotate": args.annotate,
        "max_revs": args.max_revs,
        "topk_direct": args.topk_direct,
        "topk_ga": args.topk_ga,
        "num_workers": args.num_workers,
        "thresh_kms": args.thresh_kms,
        "h_min_km": args.h_min_km,
        "h_max_km": args.h_max_km,
        "vinf_match_abs_kms": args.vinf_match_abs_kms,
        "joi_periapsis_rj": args.joi_periapsis_rj,
        "joi_apoapsis_rj": args.joi_apoapsis_rj,
    }


def make_run_output_dir():
    """Create a timestamped output folder for one solver execution."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = RESULTS_ROOT / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def serialize_traj(traj, epochs):
    """Convert a trajectory record into JSON-friendly summary data."""
    summary = {
        "type": traj["type"],
        "sequence": traj["sequence"],
        "indices": traj["indices"],
        "dates": [str(epochs[i]) for i in traj["indices"]],
        "tof_days": traj["tof_days"],
        "vinf_launch_kms": traj["vinf_launch_kms"],
        "vinf_arrive_kms": traj["vinf_arrive_kms"],
        "vinf_total_kms": traj["vinf_launch_kms"] + traj["vinf_arrive_kms"],
    }
    if "joi_delta_v_kms" in traj:
        summary["joi_delta_v_kms"] = traj["joi_delta_v_kms"]
    if "mission_total_cost_kms" in traj:
        summary["mission_total_cost_kms"] = traj["mission_total_cost_kms"]
    if "arrival_analysis" in traj:
        summary["arrival_analysis"] = traj["arrival_analysis"]
    return summary


def _format_windows(level_summary):
    """Render the searched windows for one search level as report lines."""
    lines = []
    for window in level_summary["windows"]:
        lines.append(
            f"- {window['source']}: {window['start']} to {window['stop']} "
            f"at {window['step_days']} day cadence"
        )
    return lines


def _format_trajectory_section(title, traj_summary):
    """Render one optimal-trajectory block for the mission design report."""
    total_tof_days = sum(traj_summary["tof_days"])
    lines = [
        f"## {title}",
        "",
        f"- Type: {traj_summary['type']}",
        f"- Sequence: {' -> '.join(traj_summary['sequence'])}",
        f"- Dates: {' -> '.join(traj_summary['dates'])}",
        f"- Leg TOFs (days): {', '.join(f'{tof:.1f}' for tof in traj_summary['tof_days'])}",
        f"- Total TOF (days): {total_tof_days:.1f}",
        f"- Launch v_inf (km/s): {traj_summary['vinf_launch_kms']:.3f}",
        f"- Arrival v_inf (km/s): {traj_summary['vinf_arrive_kms']:.3f}",
        f"- Total v_inf (km/s): {traj_summary['vinf_total_kms']:.3f}",
    ]
    if "joi_delta_v_kms" in traj_summary:
        lines.extend(
            [
                f"- JOI delta-v (km/s): {traj_summary['joi_delta_v_kms']:.3f}",
                f"- Launch + JOI cost (km/s): {traj_summary['mission_total_cost_kms']:.3f}",
            ]
        )
    arrival_analysis = traj_summary.get("arrival_analysis")
    if arrival_analysis:
        capture_orbit = arrival_analysis["capture_orbit"]
        lines.extend(
            [
                f"- Jupiter-relative arrival v_inf (km/s): {arrival_analysis['arrival_vinf_kms']:.3f}",
                f"- JOI delta-v (km/s): {arrival_analysis['joi_delta_v_kms']:.3f}",
                f"- Hyperbolic periapsis speed (km/s): {arrival_analysis['hyperbolic_periapsis_speed_kms']:.3f}",
                f"- Capture-orbit periapsis speed (km/s): {arrival_analysis['capture_periapsis_speed_kms']:.3f}",
                f"- Capture periapsis (Rj): {capture_orbit['periapsis_radius_rj']:.3f}",
                f"- Capture apoapsis (Rj): {capture_orbit['apoapsis_radius_rj']:.3f}",
                f"- Capture periapsis altitude (km): {capture_orbit['periapsis_radius_km'] - c.R_JUPITER:.1f}",
            ]
        )
    lines.append("")
    return lines


def write_mission_design_report(output_dir, summary):
    """Write a human-readable mission design report into the run output folder."""
    report_lines = [
        "# Mission Design Report",
        "",
        "## Search Overview",
        "",
        f"- Global search window: {summary['config']['start']} to {summary['config']['stop']}",
        f"- Initial cadence (days): {summary['config']['step_days']}",
        f"- Refinement cadences (days): {', '.join(str(step) for step in summary['config']['refine_steps']) or 'None'}",
        f"- Maximum mission duration (years): {summary['config']['max_years']}",
        f"- Lambert worker count: {summary['config']['num_workers']}",
        f"- Jupiter capture orbit: {summary['capture_model']['periapsis_radius_rj']:.1f} Rj x {summary['capture_model']['apoapsis_radius_rj']:.1f} Rj",
        "",
        "## Searched Windows",
        "",
    ]

    for level_summary in summary["search_levels"]:
        report_lines.append(
            f"### Level {level_summary['level']} ({level_summary['step_days']} day cadence)"
        )
        report_lines.append("")
        report_lines.extend(_format_windows(level_summary))
        report_lines.append("")

    report_lines.extend(
        _format_trajectory_section("Optimal Total v_inf Trajectory", summary["best_total"])
    )
    report_lines.extend(
        _format_trajectory_section("Optimal Launch + JOI Trajectory", summary["best_mission"])
    )
    report_lines.extend(
        _format_trajectory_section("Optimal Launch v_inf Trajectory", summary["best_launch"])
    )
    report_lines.extend(
        _format_trajectory_section("Optimal Arrival v_inf Trajectory", summary["best_arrival"])
    )

    (output_dir / "mission_design_report.md").write_text("\n".join(report_lines))


def _summary_payload(
    best_total_entry,
    best_mission_entry,
    best_launch_entry,
    best_arrival_entry,
    level_summaries,
    plot_result,
    window_info,
    run_config,
    capture_orbit,
):
    """Assemble the JSON summary written into the run output folder."""
    return {
        "config": run_config,
        "capture_model": jc.capture_model_summary(capture_orbit),
        "search_levels": level_summaries,
        "final_plot_window": refinement.serialize_window(plot_result["window"]),
        "stored_trajectories_in_plot_window": len(plot_result["stored"]),
        "annotated_window": (
            {
                "window_start": refinement.datetime_to_str(window_info["window_start"]),
                "window_end": refinement.datetime_to_str(window_info["window_end"]),
                "best_launch": str(window_info["best_launch"]),
                "best_arrival": str(window_info["best_arrival"]),
            }
            if window_info
            else None
        ),
        "best_total": serialize_traj(best_total_entry["traj"], best_total_entry["epochs"]),
        "best_mission": serialize_traj(best_mission_entry["traj"], best_mission_entry["epochs"]),
        "best_launch": serialize_traj(best_launch_entry["traj"], best_launch_entry["epochs"]),
        "best_arrival": serialize_traj(best_arrival_entry["traj"], best_arrival_entry["epochs"]),
    }


def _representative_entries_by_class(best_by_metric):
    """Choose one winning best-entry representative for each final trajectory class."""
    entries_by_class = {}
    for metric in ["total", "mission", "launch", "arrival"]:
        entry = best_by_metric.get(metric)
        if entry is None:
            continue
        class_name = plots.traj_class(entry["traj"])
        current = entries_by_class.get(class_name)
        if current is None or refinement.traj_total_vinf(entry["traj"]) < refinement.traj_total_vinf(current["traj"]):
            entries_by_class[class_name] = entry
    return entries_by_class


def run(args):
    """Run the full search, refinement, reporting, and plotting pipeline."""
    ls.THRESH_KMS = args.thresh_kms
    c.H_MIN = args.h_min_km
    c.H_MAX = args.h_max_km
    c.VINF_MATCH_ABS_KMS = args.vinf_match_abs_kms
    args.num_workers = ls.resolve_num_workers(args.num_workers)
    args.capture_orbit = jc.build_capture_orbit(
        args.joi_periapsis_rj, args.joi_apoapsis_rj
    )

    run_config = build_run_config(args)
    output_dir = make_run_output_dir()
    print(f"Writing results to: {output_dir}")
    print(f"Lambert precompute workers: {args.num_workers}")

    def _on_level_complete(level_index, level_results, _aggregate_best):
        if level_index != 0 or not level_results:
            return
        print("Writing level-1 porkchop plots...")
        plots.make_plots(level_results[0]["stored"], level_results[0]["epochs"], output_dir=output_dir)

    final_results, best_by_metric, level_summaries = refinement.execute_adaptive_search(
        args, serialize_traj, on_level_complete=_on_level_complete
    )
    best_total_entry = best_by_metric["total"]
    best_mission_entry = best_by_metric["mission"]
    best_launch_entry = best_by_metric["launch"]
    best_arrival_entry = best_by_metric["arrival"]
    if best_total_entry is None:
        raise RuntimeError("No trajectories were found in the searched windows.")

    plot_result = {
        "window": best_total_entry["window"],
        "stored": best_total_entry["stored"],
    }
    window_info = refinement.make_window_info(best_total_entry) if args.annotate else None
    annotated_entries_by_class = (
        _representative_entries_by_class(best_by_metric) if args.annotate else {}
    )

    print("\n\n")
    print_solution_summary("Best total v_inf trajectory", best_total_entry["traj"], best_total_entry["epochs"])
    print_solution_summary("Best launch + JOI trajectory", best_mission_entry["traj"], best_mission_entry["epochs"])
    print_solution_summary("Best launch v_inf trajectory", best_launch_entry["traj"], best_launch_entry["epochs"])
    print_solution_summary("Best arrival v_inf trajectory", best_arrival_entry["traj"], best_arrival_entry["epochs"])

    summary = _summary_payload(
        best_total_entry,
        best_mission_entry,
        best_launch_entry,
        best_arrival_entry,
        level_summaries,
        plot_result,
        window_info,
        run_config,
        args.capture_orbit,
    )
    (output_dir / "run_config.json").write_text(json.dumps(summary, indent=2))
    write_mission_design_report(output_dir, summary)
    snapshot_io.write_trajectory_snapshot(
        output_dir,
        {
            "total": best_total_entry,
            "mission": best_mission_entry,
            "launch": best_launch_entry,
            "arrival": best_arrival_entry,
        },
        summary,
        serialize_traj,
    )

    if annotated_entries_by_class:
        print("Writing final annotated porkchop plots...")
        plots.make_annotated_plots(annotated_entries_by_class, output_dir=output_dir)
    plots.plot_spacecraft_traj(
        best_total_entry["traj"],
        best_total_entry["epochs"],
        best_total_entry["bodies"],
        tag="total",
        output_dir=output_dir,
    )
    plots.plot_spacecraft_traj(
        best_mission_entry["traj"],
        best_mission_entry["epochs"],
        best_mission_entry["bodies"],
        tag="mission",
        output_dir=output_dir,
    )
    plots.plot_spacecraft_traj(
        best_launch_entry["traj"],
        best_launch_entry["epochs"],
        best_launch_entry["bodies"],
        tag="launch",
        output_dir=output_dir,
    )
    plots.plot_spacecraft_traj(
        best_arrival_entry["traj"],
        best_arrival_entry["epochs"],
        best_arrival_entry["bodies"],
        tag="arrival",
        output_dir=output_dir,
    )


def main(argv=None):
    """Parse CLI arguments and execute the configured search run."""
    run(parse_args(argv))


if __name__ == "__main__":
    mp.freeze_support()
    main()
