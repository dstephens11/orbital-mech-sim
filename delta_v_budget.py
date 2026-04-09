"""Build a delta-v budget from a saved trajectory snapshot."""

import argparse
import json
from pathlib import Path

from arrival import earth_departure as ed
import constants as c
import snapshot_io

DEFAULT_RESULTS_ROOT = Path("results")
DEFAULT_MARGIN_PCT = 15.0
DEFAULT_POST_LAUNCH_TCM_MS = 30.0
DEFAULT_PRE_FLYBY_TCM_MS = 20.0
DEFAULT_JUPITER_CRUISE_TCM_MS_PER_YEAR = 15.0
DEFAULT_POST_JOI_TCM_FRAC = 0.02
DEFAULT_POST_JOI_TCM_MIN_MS = 10.0
DEFAULT_JUPITER_OPS_YEARS = 5.0
DEFAULT_STATIONKEEPING_MS_PER_YEAR = 20.0
DEFAULT_DISPOSAL_PERIAPSIS_RJ = 1.0


def parse_args(argv=None):
    """Parse command-line arguments for downstream mission delta-v budgeting."""
    parser = argparse.ArgumentParser(
        description=(
            "Create a mission delta-v budget from a saved trajectory snapshot. "
            "Defaults to the most recent results folder and the saved best mission trajectory."
        )
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help=(
            "Path to a results folder or trajectory_ephemeris_snapshot.json. "
            "Defaults to the latest folder under results/."
        ),
    )
    parser.add_argument(
        "--best",
        choices=snapshot_io.SNAPSHOT_BEST_LABELS,
        default="mission",
        help="Which saved best-trajectory family to budget.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for the budget outputs. Defaults to the selected results folder.",
    )
    parser.add_argument(
        "--margin-pct",
        type=float,
        default=DEFAULT_MARGIN_PCT,
        help="Flat margin percentage applied to the budget.",
    )
    parser.add_argument(
        "--post-launch-tcm-ms",
        type=float,
        default=DEFAULT_POST_LAUNCH_TCM_MS,
        help="Allocation for post-launch correction maneuvers in m/s.",
    )
    parser.add_argument(
        "--pre-flyby-tcm-ms",
        type=float,
        default=DEFAULT_PRE_FLYBY_TCM_MS,
        help="Allocation per flyby for pre-flyby cleanup maneuvers in m/s.",
    )
    parser.add_argument(
        "--jupiter-cruise-tcm-ms-per-year",
        type=float,
        default=DEFAULT_JUPITER_CRUISE_TCM_MS_PER_YEAR,
        help="Allocation rate for cruise correction maneuvers on the final Jupiter leg in m/s per year.",
    )
    parser.add_argument(
        "--post-joi-tcm-frac",
        type=float,
        default=DEFAULT_POST_JOI_TCM_FRAC,
        help="Fraction of the JOI burn allocated to post-JOI orbit-trim cleanup maneuvers.",
    )
    parser.add_argument(
        "--post-joi-tcm-min-ms",
        type=float,
        default=DEFAULT_POST_JOI_TCM_MIN_MS,
        help="Minimum allocation for post-JOI cleanup maneuvers in m/s.",
    )
    parser.add_argument(
        "--jupiter-ops-years",
        type=float,
        default=DEFAULT_JUPITER_OPS_YEARS,
        help="Assumed time spent in Jupiter orbit for station-keeping allocation.",
    )
    parser.add_argument(
        "--stationkeeping-ms-per-year",
        type=float,
        default=DEFAULT_STATIONKEEPING_MS_PER_YEAR,
        help="Station-keeping allocation rate after JOI in m/s per year.",
    )
    parser.add_argument(
        "--disposal-periapsis-rj",
        type=float,
        default=DEFAULT_DISPOSAL_PERIAPSIS_RJ,
        help=(
            "Target periapsis radius in Jupiter radii for the end-of-mission disposal burn. "
            "If this is greater than or equal to the captured-orbit periapsis, the disposal burn is zero."
        ),
    )
    return parser.parse_args(argv)


def _latest_results_dir(results_root=DEFAULT_RESULTS_ROOT):
    """Return the newest timestamped results directory."""
    results_root = Path(results_root)
    if not results_root.exists():
        raise FileNotFoundError(f"Results root not found: {results_root}")

    candidates = sorted(path for path in results_root.iterdir() if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No results folders found under: {results_root}")
    return candidates[-1]


def _resolve_results_context(path_arg):
    """Resolve the selected results directory and snapshot path from one user path."""
    if path_arg is None:
        results_dir = _latest_results_dir().resolve()
        snapshot_path = results_dir / "trajectory_ephemeris_snapshot.json"
        return results_dir, snapshot_path

    input_path = Path(path_arg).expanduser().resolve()
    if input_path.is_dir():
        results_dir = input_path
        snapshot_path = results_dir / "trajectory_ephemeris_snapshot.json"
        return results_dir, snapshot_path

    if input_path.name == "trajectory_ephemeris_snapshot.json":
        return input_path.parent, input_path

    raise FileNotFoundError(
        "Path must point to a results folder or trajectory_ephemeris_snapshot.json."
    )


def _years_from_days(days):
    """Convert a duration in days to Julian years."""
    return float(days) / 365.25


def _ms_from_kms(value_kms):
    """Convert km/s to m/s."""
    return 1000.0 * float(value_kms)


def launch_analysis_to_ms(launch_analysis):
    """Convert launch-analysis speed terms to m/s while keeping C3 in standard units."""
    return {
        "model": launch_analysis["model"],
        "launch_site": launch_analysis["launch_site"],
        "launch_latitude_deg": float(launch_analysis["launch_latitude_deg"]),
        "parking_altitude_km": float(launch_analysis["parking_altitude_km"]),
        "parking_radius_km": float(launch_analysis["parking_radius_km"]),
        "vinf_launch_ms": _ms_from_kms(launch_analysis["vinf_launch_kms"]),
        "c3_km2_s2": float(launch_analysis["c3_km2_s2"]),
        "parking_circular_speed_ms": _ms_from_kms(
            launch_analysis["parking_circular_speed_kms"]
        ),
        "hyperbolic_perigee_speed_ms": _ms_from_kms(
            launch_analysis["hyperbolic_perigee_speed_kms"]
        ),
        "escape_burn_delta_v_ms": _ms_from_kms(
            launch_analysis["escape_burn_delta_v_kms"]
        ),
        "earth_rotation_benefit_ms": _ms_from_kms(
            launch_analysis["earth_rotation_benefit_kms"]
        ),
        "notes": launch_analysis["notes"],
    }


def _capture_disposal_delta_v_ms(capture_orbit, disposal_periapsis_rj):
    """Estimate the apoapsis burn needed to lower periapsis into Jupiter."""
    current_rp_km = float(capture_orbit["periapsis_radius_km"])
    target_rp_km = float(disposal_periapsis_rj) * c.R_JUPITER
    ra_km = float(capture_orbit["apoapsis_radius_km"])

    if target_rp_km >= current_rp_km:
        return 0.0

    a_current_km = 0.5 * (current_rp_km + ra_km)
    a_target_km = 0.5 * (target_rp_km + ra_km)
    v_apo_current_kms = (c.MU_JUPITER * (2.0 / ra_km - 1.0 / a_current_km)) ** 0.5
    v_apo_target_kms = (c.MU_JUPITER * (2.0 / ra_km - 1.0 / a_target_km)) ** 0.5
    return max(0.0, _ms_from_kms(v_apo_current_kms - v_apo_target_kms))


def _build_budget_item(name, maneuver_class, base_delta_v_ms, margin_fraction, notes):
    """Create one budget line item with a class, base value, and flat margin."""
    base_delta_v_ms = float(base_delta_v_ms)
    margin_fraction = float(margin_fraction)
    margin_ms = base_delta_v_ms * margin_fraction
    return {
        "name": name,
        "maneuver_class": maneuver_class,
        "base_delta_v_ms": base_delta_v_ms,
        "margin_fraction": margin_fraction,
        "margin_ms": margin_ms,
        "budget_delta_v_ms": base_delta_v_ms + margin_ms,
        "notes": notes,
    }


def build_delta_v_budget(trajectory, args):
    """Build the mission delta-v budget line items for one selected trajectory."""
    margin_fraction = args.margin_pct / 100.0
    flybys = trajectory.get("flyby", [])
    flyby_count = len(flybys)
    flyby_bodies = [flyby["body"] for flyby in flybys]
    final_leg_days = float(trajectory["tof_days"][-1])
    final_leg_years = _years_from_days(final_leg_days)

    if "arrival_analysis" not in trajectory:
        raise ValueError(
            "Selected trajectory is missing arrival_analysis, so JOI and post-JOI budget terms cannot be estimated."
        )

    arrival_analysis = trajectory["arrival_analysis"]
    launch_analysis = trajectory.get("launch_analysis")
    if launch_analysis is None:
        launch_analysis = ed.compute_earth_departure(trajectory["vinf_launch_kms"])
    joi_delta_v_ms = _ms_from_kms(arrival_analysis["joi_delta_v_kms"])
    post_joi_trim_ms = max(
        float(args.post_joi_tcm_min_ms), joi_delta_v_ms * float(args.post_joi_tcm_frac)
    )
    stationkeeping_ms = float(args.jupiter_ops_years) * float(
        args.stationkeeping_ms_per_year
    )
    disposal_delta_v_ms = _capture_disposal_delta_v_ms(
        arrival_analysis["capture_orbit"], args.disposal_periapsis_rj
    )

    items = [
        _build_budget_item(
            "Launch",
            "deterministic",
            _ms_from_kms(launch_analysis["escape_burn_delta_v_kms"]),
            margin_fraction,
            (
                f"Patched-conic Earth departure from a {launch_analysis['parking_altitude_km']:.0f} km "
                f"circular parking orbit at {launch_analysis['launch_site']}. "
                f"Required C3 = {launch_analysis['c3_km2_s2']:.1f} km^2/s^2."
            ),
        ),
        _build_budget_item(
            "Corrections After Launch",
            "statistical",
            args.post_launch_tcm_ms,
            margin_fraction,
            "Early trajectory correction and injection cleanup allowance.",
        ),
        _build_budget_item(
            "Corrections Before Flybys",
            "statistical",
            flyby_count * float(args.pre_flyby_tcm_ms),
            margin_fraction,
            (
                f"Allocates {args.pre_flyby_tcm_ms:.1f} m/s before each flyby "
                f"({flyby_count} total: {', '.join(flyby_bodies) if flyby_bodies else 'none'})."
            ),
        ),
        _build_budget_item(
            "Corrections On The Way To Jupiter",
            "statistical",
            final_leg_years * float(args.jupiter_cruise_tcm_ms_per_year),
            margin_fraction,
            (
                f"Allocates {args.jupiter_cruise_tcm_ms_per_year:.1f} m/s/year "
                f"across the final Jupiter transfer leg ({final_leg_days:.1f} days)."
            ),
        ),
        _build_budget_item(
            "JOI Burn",
            "deterministic",
            joi_delta_v_ms,
            margin_fraction,
            "Impulsive Jupiter orbit insertion from the arrival vector.",
        ),
        _build_budget_item(
            "Corrections After JOI Burn",
            "statistical",
            post_joi_trim_ms,
            margin_fraction,
            (
                f"Uses max({args.post_joi_tcm_min_ms:.1f} m/s, "
                f"{args.post_joi_tcm_frac*100:.1f}% of JOI) for capture-orbit trim and cleanup."
            ),
        ),
        _build_budget_item(
            "Station-Keeping After JOI",
            "statistical",
            stationkeeping_ms,
            margin_fraction,
            (
                f"Allocates {args.stationkeeping_ms_per_year:.1f} m/s/year over "
                f"{args.jupiter_ops_years:.2f} years of Jupiter-orbit operations."
            ),
        ),
        _build_budget_item(
            "End-Of-Mission Jupiter Disposal",
            "deterministic",
            disposal_delta_v_ms,
            margin_fraction,
            (
                f"Estimated apoapsis burn to lower periapsis to {args.disposal_periapsis_rj:.1f} Rj "
                "for final Jupiter impact/disposal."
            ),
        ),
    ]

    non_launch_items = [item for item in items if item["name"] != "Launch"]

    deterministic_total_ms = sum(
        item["base_delta_v_ms"]
        for item in items
        if item["maneuver_class"] == "deterministic"
    )
    statistical_total_ms = sum(
        item["base_delta_v_ms"]
        for item in items
        if item["maneuver_class"] == "statistical"
    )
    margin_total_ms = sum(item["margin_ms"] for item in items)
    total_budget_ms = sum(item["budget_delta_v_ms"] for item in items)

    deterministic_total_without_launch_ms = sum(
        item["base_delta_v_ms"]
        for item in non_launch_items
        if item["maneuver_class"] == "deterministic"
    )
    statistical_total_without_launch_ms = sum(
        item["base_delta_v_ms"]
        for item in non_launch_items
        if item["maneuver_class"] == "statistical"
    )
    margin_total_without_launch_ms = sum(item["margin_ms"] for item in non_launch_items)
    total_budget_without_launch_ms = sum(
        item["budget_delta_v_ms"] for item in non_launch_items
    )

    assumptions = {
        "margin_pct": float(args.margin_pct),
        "post_launch_tcm_ms": float(args.post_launch_tcm_ms),
        "pre_flyby_tcm_ms": float(args.pre_flyby_tcm_ms),
        "jupiter_cruise_tcm_ms_per_year": float(args.jupiter_cruise_tcm_ms_per_year),
        "post_joi_tcm_frac": float(args.post_joi_tcm_frac),
        "post_joi_tcm_min_ms": float(args.post_joi_tcm_min_ms),
        "jupiter_ops_years": float(args.jupiter_ops_years),
        "stationkeeping_ms_per_year": float(args.stationkeeping_ms_per_year),
        "disposal_periapsis_rj": float(args.disposal_periapsis_rj),
    }

    return {
        "assumptions": assumptions,
        "items": items,
        "totals": {
            "deterministic_delta_v_ms": deterministic_total_ms,
            "statistical_delta_v_ms": statistical_total_ms,
            "margin_delta_v_ms": margin_total_ms,
            "total_budget_delta_v_ms": total_budget_ms,
            "deterministic_delta_v_without_launch_ms": deterministic_total_without_launch_ms,
            "statistical_delta_v_without_launch_ms": statistical_total_without_launch_ms,
            "margin_delta_v_without_launch_ms": margin_total_without_launch_ms,
            "total_budget_delta_v_without_launch_ms": total_budget_without_launch_ms,
        },
    }


def build_budget_payload(results_dir, snapshot_path, best_label, best_payload, args):
    """Build the JSON payload written by the delta-v budgeting tool."""
    trajectory = best_payload["trajectory"]
    budget = build_delta_v_budget(trajectory, args)
    return {
        "results_dir": str(results_dir),
        "snapshot_path": str(snapshot_path),
        "selected_best": best_label,
        "selected_window": best_payload["window"],
        "trajectory": {
            "type": trajectory["type"],
            "sequence": trajectory["sequence"],
            "dates": trajectory["dates"],
            "tof_days": trajectory["tof_days"],
            "launch_vinf_ms": _ms_from_kms(trajectory["vinf_launch_kms"]),
            "jupiter_arrival_vinf_ms": _ms_from_kms(trajectory["vinf_arrive_kms"]),
            "joi_delta_v_ms": (
                None
                if trajectory.get("joi_delta_v_kms") is None
                else _ms_from_kms(trajectory["joi_delta_v_kms"])
            ),
            "launch_analysis": launch_analysis_to_ms(
                trajectory.get("launch_analysis")
                if trajectory.get("launch_analysis") is not None
                else ed.compute_earth_departure(trajectory["vinf_launch_kms"])
            ),
            "mission_total_cost_proxy_ms": (
                None
                if trajectory.get("mission_total_cost_kms") is None
                else _ms_from_kms(trajectory["mission_total_cost_kms"])
            ),
        },
        "budget": budget,
    }


def render_budget_report(payload):
    """Render a human-readable Markdown mission budget report."""
    trajectory = payload["trajectory"]
    budget = payload["budget"]

    launch_analysis = trajectory["launch_analysis"]
    lines = [
        "# Mission Delta-v Budget",
        "",
        "## Selected Trajectory",
        "",
        f"- Selected best trajectory: {payload['selected_best']}",
        f"- Type: {trajectory['type']}",
        f"- Sequence: {' -> '.join(trajectory['sequence'])}",
        f"- Dates: {' -> '.join(date.split()[0] for date in trajectory['dates'])}",
        f"- Leg TOFs (days): {', '.join(f'{tof:.1f}' for tof in trajectory['tof_days'])}",
        f"- Launch v_inf (m/s): {trajectory['launch_vinf_ms']:.1f}",
        f"- Launch C3 (km^2/s^2): {launch_analysis['c3_km2_s2']:.3f}",
        f"- Launch escape burn delta-v (m/s): {launch_analysis['escape_burn_delta_v_ms']:.1f}",
        f"- Launch site: {launch_analysis['launch_site']}",
        f"- Ideal Earth-rotation assist at site (m/s): {launch_analysis['earth_rotation_benefit_ms']:.1f}",
        f"- Jupiter arrival v_inf (m/s): {trajectory['jupiter_arrival_vinf_ms']:.1f}",
    ]

    if trajectory["joi_delta_v_ms"] is not None:
        lines.append(f"- JOI delta-v (m/s): {trajectory['joi_delta_v_ms']:.1f}")
    if trajectory["mission_total_cost_proxy_ms"] is not None:
        lines.append(
            f"- Launch + JOI search metric proxy (m/s): {trajectory['mission_total_cost_proxy_ms']:.1f}"
        )

    assumptions = budget["assumptions"]
    lines.extend(
        [
            "",
            "## Budget Assumptions",
            "",
            f"- Flat margin: {assumptions['margin_pct']:.1f}%",
            f"- Post-launch corrections (m/s): {assumptions['post_launch_tcm_ms']:.1f}",
            f"- Pre-flyby allowance per flyby (m/s): {assumptions['pre_flyby_tcm_ms']:.1f}",
            f"- Jupiter-leg cruise corrections (m/s/year): {assumptions['jupiter_cruise_tcm_ms_per_year']:.1f}",
            f"- Post-JOI cleanup floor (m/s): {assumptions['post_joi_tcm_min_ms']:.1f}",
            f"- Post-JOI cleanup fraction of JOI: {assumptions['post_joi_tcm_frac']:.3f}",
            f"- Jupiter operations duration (years): {assumptions['jupiter_ops_years']:.2f}",
            f"- Station-keeping rate (m/s/year): {assumptions['stationkeeping_ms_per_year']:.1f}",
            f"- End-of-mission disposal periapsis target (Rj): {assumptions['disposal_periapsis_rj']:.3f}",
            "",
            "## Budget Table",
            "",
            "| Mission Element | Maneuver Class | Base (m/s) | Margin (m/s) | Element Budget (m/s) | Notes |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )

    for item in budget["items"]:
        lines.append(
            f"| {item['name']} | {item['maneuver_class']} | {item['base_delta_v_ms']:.1f} | "
            f"{item['margin_ms']:.1f} | {item['budget_delta_v_ms']:.1f} | "
            f"{item['notes']} |"
        )

    totals = budget["totals"]
    lines.extend(
        [
            "| **Budget Summary** |  |  |  |  |  |",
            f"| Without Launch Delta-v | combined | "
            f"{totals['deterministic_delta_v_without_launch_ms'] + totals['statistical_delta_v_without_launch_ms']:.1f} | "
            f"0.0 | {totals['deterministic_delta_v_without_launch_ms'] + totals['statistical_delta_v_without_launch_ms']:.1f} |  |",
            f"| Without Launch Total Margin | margin | 0.0 | "
            f"{totals['margin_delta_v_without_launch_ms']:.1f} | "
            f"{totals['margin_delta_v_without_launch_ms']:.1f} | |",
            f"| **Without Launch Total** | combined | "
            f"{totals['deterministic_delta_v_without_launch_ms'] + totals['statistical_delta_v_without_launch_ms']:.1f} | "
            f"{totals['margin_delta_v_without_launch_ms']:.1f} | "
            f"{totals['total_budget_delta_v_without_launch_ms']:.1f} | "
            "Total budget excluding launch. |",
            f"| With Launch Delta-v | combined | "
            f"{totals['deterministic_delta_v_ms'] + totals['statistical_delta_v_ms']:.1f} | "
            f"0.0 | {totals['deterministic_delta_v_ms'] + totals['statistical_delta_v_ms']:.1f} | |",
            f"| With Launch Total Margin | margin | 0.0 | "
            f"{totals['margin_delta_v_ms']:.1f} | "
            f"{totals['margin_delta_v_ms']:.1f} | |",
            f"| **With Launch Total** | combined | "
            f"{totals['deterministic_delta_v_ms'] + totals['statistical_delta_v_ms']:.1f} | "
            f"{totals['margin_delta_v_ms']:.1f} | "
            f"{totals['total_budget_delta_v_ms']:.1f} | "
            "Total budget including launch. |",
            "",
            "## Notes",
            "",
            "- Deterministic items are explicit planned maneuvers tied to the selected trajectory or disposal strategy.",
            "- Statistical items are planning allowances for cleanup.",
            "- The launch term now uses a patched-conic escape burn from circular Earth parking orbit.",
            "- The Earth-rotation term is reported separately as an ideal launch-site assist and is not subtracted from the parking-orbit escape burn.",
            "",
        ]
    )
    return lines


def main(argv=None):
    """Create a mission delta-v budget from a saved results folder."""
    args = parse_args(argv)
    results_dir, snapshot_path = _resolve_results_context(args.path)
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")

    output_dir = (
        results_dir
        if args.output_dir is None
        else Path(args.output_dir).expanduser().resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot = snapshot_io.load_snapshot(snapshot_path)
    best_payload = snapshot_io.get_snapshot_best(snapshot, args.best)
    payload = build_budget_payload(
        results_dir, snapshot_path, args.best, best_payload, args
    )

    outfile_stem = f"delta_v_budget_{args.best}"
    json_outfile = output_dir / f"{outfile_stem}.json"
    md_outfile = output_dir / f"{outfile_stem}.md"
    json_outfile.write_text(json.dumps(payload, indent=2))
    md_outfile.write_text("\n".join(render_budget_report(payload)))

    print(f"Results folder: {results_dir}")
    print(f"Snapshot: {snapshot_path}")
    print(f"Selected best trajectory: {args.best}")
    print(
        f"Final budget delta-v total: {payload['budget']['totals']['total_budget_delta_v_ms']:.1f} m/s"
    )
    print(
        "Final budget delta-v total without launch: "
        f"{payload['budget']['totals']['total_budget_delta_v_without_launch_ms']:.1f} m/s"
    )
    print(f"Wrote: {json_outfile}")
    print(f"Wrote: {md_outfile}")


if __name__ == "__main__":
    main()
