"""Re-run Jupiter capture analysis from a saved trajectory snapshot."""

import argparse
import json
from pathlib import Path

import constants as c
import numpy as np
from arrival import jupiter_capture as jc
import snapshot_io
from visualization import capture as capture_viz


def parse_args(argv=None):
    """Parse command-line settings for snapshot-driven capture reanalysis."""
    parser = argparse.ArgumentParser(
        description="Recompute Jupiter orbit insertion from a saved trajectory snapshot."
    )
    parser.add_argument(
        "snapshot",
        help="Path to trajectory_ephemeris_snapshot.json from a prior search run.",
    )
    parser.add_argument(
        "--best",
        choices=snapshot_io.SNAPSHOT_BEST_LABELS,
        default="mission",
        help="Which saved best-trajectory family to reanalyze.",
    )
    parser.add_argument(
        "--periapsis-rj",
        type=float,
        default=c.JOI_CAPTURE_PERIAPSIS_RJ,
        help="Target capture-orbit periapsis radius in Jupiter radii.",
    )
    parser.add_argument(
        "--apoapsis-rj",
        type=float,
        default=c.JOI_CAPTURE_APOAPSIS_RJ,
        help="Target capture-orbit apoapsis radius in Jupiter radii.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for the reanalysis outputs. Defaults to the snapshot folder.",
    )
    return parser.parse_args(argv)


def _arrival_state_from_snapshot(best_payload):
    """Extract saved arrival-state vectors from one snapshot trajectory."""
    trajectory = best_payload["trajectory"]
    arrival_analysis = trajectory.get("arrival_analysis")
    if arrival_analysis is None:
        raise ValueError(
            "Snapshot trajectory is missing arrival_analysis. Re-run the search with the current snapshot format."
        )
    return trajectory, arrival_analysis


def _sun_direction_from_snapshot(snapshot_path, best_label, trajectory):
    """Return the Jupiter-to-Sun XY vector at arrival, if the NPZ snapshot is available."""
    snapshot_npz_path = snapshot_path.with_suffix(".npz")
    if not snapshot_npz_path.exists():
        return None

    arrays = snapshot_io.load_snapshot_npz(snapshot_npz_path)
    key = f"{best_label}_jupiter_pos_au"
    if key not in arrays:
        return None

    arrival_index = trajectory["indices"][-1]
    jupiter_pos_au = np.asarray(arrays[key], dtype=float)
    if arrival_index < 0 or arrival_index >= len(jupiter_pos_au):
        return None

    return -jupiter_pos_au[arrival_index, :2] * c.AU_KM


def build_reanalysis_payload(
    snapshot_path,
    best_label,
    best_payload,
    baseline_capture_model,
    capture_orbit,
    analysis,
):
    """Build the JSON payload written by the standalone capture reanalysis tool."""
    trajectory = dict(best_payload["trajectory"])
    baseline_joi_delta_v_kms = trajectory.get("joi_delta_v_kms")
    baseline_mission_total_cost_kms = trajectory.get("mission_total_cost_kms")
    mission_total_cost_kms = trajectory["vinf_launch_kms"] + analysis["joi_delta_v_kms"]

    trajectory["arrival_analysis"] = analysis
    trajectory["joi_delta_v_kms"] = analysis["joi_delta_v_kms"]
    trajectory["mission_total_cost_kms"] = mission_total_cost_kms

    payload = {
        "snapshot_path": str(snapshot_path),
        "selected_best": best_label,
        "selected_window": best_payload["window"],
        "capture_model": jc.capture_model_summary(capture_orbit),
        "trajectory": trajectory,
        "baseline_capture_model": baseline_capture_model,
        "baseline_joi_delta_v_kms": baseline_joi_delta_v_kms,
        "baseline_mission_total_cost_kms": baseline_mission_total_cost_kms,
        "geometry_note": (
            "Jupiter-centered visualization is a representative planar arrival/capture view "
            "derived from the saved arrival v_inf and target capture orbit. "
            "When the sibling NPZ snapshot is available, the plot also shows the Jupiter-to-Sun "
            "direction from the saved arrival ephemeris, and the approximate capture ellipse "
            "is oriented with periapsis Sunward and apoapsis anti-Sunward by convention."
        ),
    }
    return payload


def _report_lines(payload):
    """Render a compact Markdown report for one capture reanalysis."""
    trajectory = payload["trajectory"]
    arrival_analysis = trajectory["arrival_analysis"]
    capture_orbit = arrival_analysis["capture_orbit"]
    total_tof_days = sum(trajectory["tof_days"])

    lines = [
        "# Jupiter Capture Reanalysis",
        "",
        f"- Snapshot: {payload['snapshot_path']}",
        f"- Selected best trajectory: {payload['selected_best']}",
        f"- Type: {trajectory['type']}",
        f"- Sequence: {' -> '.join(trajectory['sequence'])}",
        f"- Dates: {' -> '.join(trajectory['dates'])}",
        f"- Total TOF (days): {total_tof_days:.1f}",
        f"- Launch v_inf (km/s): {trajectory['vinf_launch_kms']:.3f}",
        f"- Arrival v_inf (km/s): {trajectory['vinf_arrive_kms']:.3f}",
        f"- Jupiter-relative arrival v_inf (km/s): {arrival_analysis['arrival_vinf_kms']:.3f}",
        f"- JOI delta-v (km/s): {arrival_analysis['joi_delta_v_kms']:.3f}",
        f"- Launch + JOI cost (km/s): {trajectory['mission_total_cost_kms']:.3f}",
        f"- Capture periapsis (Rj): {capture_orbit['periapsis_radius_rj']:.3f}",
        f"- Capture apoapsis (Rj): {capture_orbit['apoapsis_radius_rj']:.3f}",
        f"- Capture periapsis altitude (km): {capture_orbit['periapsis_radius_km'] - c.R_JUPITER:.1f}",
        f"- Geometry note: {payload['geometry_note']}",
    ]

    if payload["baseline_joi_delta_v_kms"] is not None:
        delta_joi = arrival_analysis["joi_delta_v_kms"] - payload["baseline_joi_delta_v_kms"]
        lines.append(f"- JOI delta vs saved baseline (km/s): {delta_joi:+.3f}")
    if payload["baseline_mission_total_cost_kms"] is not None:
        delta_mission = (
            trajectory["mission_total_cost_kms"] - payload["baseline_mission_total_cost_kms"]
        )
        lines.append(f"- Launch + JOI vs saved baseline (km/s): {delta_mission:+.3f}")

    lines.append("")
    return lines


def main(argv=None):
    """Load a saved snapshot trajectory and re-run Jupiter capture analysis."""
    args = parse_args(argv)
    snapshot_path = Path(args.snapshot).expanduser().resolve()
    output_dir = (
        snapshot_path.parent if args.output_dir is None else Path(args.output_dir).expanduser().resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot = snapshot_io.load_snapshot(snapshot_path)
    best_payload = snapshot_io.get_snapshot_best(snapshot, args.best)
    trajectory, saved_arrival_analysis = _arrival_state_from_snapshot(best_payload)

    capture_orbit = jc.build_capture_orbit(args.periapsis_rj, args.apoapsis_rj)
    analysis = jc.compute_jupiter_capture(
        saved_arrival_analysis["spacecraft_arrival_velocity_kms"],
        saved_arrival_analysis["jupiter_arrival_velocity_kms"],
        capture_orbit=capture_orbit,
    )
    sun_direction_xy_km = _sun_direction_from_snapshot(snapshot_path, args.best, trajectory)
    payload = build_reanalysis_payload(
        snapshot_path,
        args.best,
        best_payload,
        snapshot.get("capture_model"),
        capture_orbit,
        analysis,
    )

    outfile_stem = f"jupiter_capture_{args.best}"
    (output_dir / f"{outfile_stem}.json").write_text(json.dumps(payload, indent=2))
    (output_dir / f"{outfile_stem}.md").write_text("\n".join(_report_lines(payload)))
    plot_outfile, animation_outfile = capture_viz.plot_jupiter_capture(
        payload["trajectory"]["arrival_analysis"],
        tag=args.best,
        output_dir=output_dir,
        sun_direction_xy_km=sun_direction_xy_km,
    )

    print(f"Snapshot: {snapshot_path}")
    print(f"Selected best trajectory: {args.best}")
    print(
        f"Capture orbit: {capture_orbit['periapsis_radius_rj']:.3f} Rj x "
        f"{capture_orbit['apoapsis_radius_rj']:.3f} Rj"
    )
    print(f"JOI delta-v: {analysis['joi_delta_v_kms']:.3f} km/s")
    print(
        f"Launch + JOI cost: {payload['trajectory']['mission_total_cost_kms']:.3f} km/s"
    )
    print(f"Wrote: {output_dir / f'{outfile_stem}.json'}")
    print(f"Wrote: {output_dir / f'{outfile_stem}.md'}")
    print(f"Wrote: {plot_outfile}")
    print(f"Wrote: {animation_outfile}")


if __name__ == "__main__":
    main()
