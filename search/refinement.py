"""Adaptive coarse-to-fine search helpers."""

from datetime import datetime, timedelta

import numpy as np

import ephemeris as hr
from search import lambert as ls


def parse_csv_ints(raw_value):
    """Parse a comma-separated list of integer cadence values."""
    if raw_value is None:
        return None

    values = []
    for chunk in raw_value.split(","):
        chunk = chunk.strip()
        if chunk:
            values.append(int(chunk))
    return values


def default_refine_steps(base_step):
    """Build a default refinement ladder by halving cadence down to one day."""
    steps = []
    current = max(1, int(base_step))
    while current > 1:
        current = max(1, current // 2)
        if current not in steps:
            steps.append(current)
        if current == 1:
            break
    return steps


def datetime_to_str(value):
    """Format a ``datetime`` as the YYYY-MM-DD string expected by Horizons."""
    return value.strftime("%Y-%m-%d")


def traj_total_vinf(traj):
    """Return the total launch-plus-arrival ``v_inf`` used for ranking."""
    return traj["vinf_launch_kms"] + traj["vinf_arrive_kms"]


def traj_mission_cost(traj):
    """Return the launch-plus-JOI mission cost used for ranking."""
    return traj["mission_total_cost_kms"]


def make_best_entry(traj, epochs, bodies, window, stored):
    """Bundle a best trajectory with the context needed to use it later."""
    return {
        "traj": traj,
        "epochs": epochs,
        "bodies": bodies,
        "window": window,
        "stored": stored,
    }


def pick_better_best(current_entry, candidate_traj, epochs, bodies, window, stored, key):
    """Compare a candidate against the current best for one ranking metric."""
    if candidate_traj is None:
        return current_entry

    if current_entry is None:
        return make_best_entry(candidate_traj, epochs, bodies, window, stored)

    current_traj = current_entry["traj"]
    if key == "launch":
        is_better = candidate_traj["vinf_launch_kms"] < current_traj["vinf_launch_kms"]
    elif key == "arrival":
        is_better = candidate_traj["vinf_arrive_kms"] < current_traj["vinf_arrive_kms"]
    elif key == "mission":
        is_better = traj_mission_cost(candidate_traj) < traj_mission_cost(current_traj)
    else:
        is_better = traj_total_vinf(candidate_traj) < traj_total_vinf(current_traj)

    if is_better:
        return make_best_entry(candidate_traj, epochs, bodies, window, stored)
    return current_entry


def serialize_window(window):
    """Convert a search window definition into JSON-friendly data."""
    return {
        "start": datetime_to_str(window["start"]),
        "stop": datetime_to_str(window["stop"]),
        "step_days": window["step"],
        "source": window.get("source", "manual"),
    }


def summarize_ranked_trajectories(ranked, epochs, window):
    """Attach dates and scores to ranked trajectories for corridor selection."""
    summaries = []
    for traj in ranked:
        summaries.append(
            {
                "traj": traj,
                "launch_date": epochs[traj["indices"][0]],
                "arrival_date": epochs[traj["indices"][-1]],
                "window": window,
                "score_total": traj_total_vinf(traj),
            }
        )
    return summaries


def select_refinement_windows(
    candidate_summaries,
    next_step,
    top_n,
    pad_scale,
    global_start,
    global_stop,
):
    """Choose the next set of refined search windows from top candidate corridors."""
    pad_days = max(next_step, int(np.ceil(next_step * pad_scale)))
    selected = []

    for candidate in sorted(candidate_summaries, key=lambda item: item["score_total"]):
        launch_date = candidate["launch_date"]
        arrival_date = candidate["arrival_date"]

        if any(
            abs((launch_date - existing["seed_launch"]).days) <= pad_days
            and abs((arrival_date - existing["seed_arrival"]).days) <= pad_days
            for existing in selected
        ):
            continue

        pad = timedelta(days=pad_days)
        start = max(global_start, launch_date - pad)
        stop = min(global_stop, arrival_date + pad)
        if stop <= start:
            continue

        selected.append(
            {
                "start": start,
                "stop": stop,
                "step": next_step,
                "source": "adaptive_refinement",
                "seed_launch": launch_date,
                "seed_arrival": arrival_date,
            }
        )
        if len(selected) >= top_n:
            break

    for window in selected:
        window.pop("seed_launch", None)
        window.pop("seed_arrival", None)

    return selected


def run_search_window(args, window):
    """Execute one search window and package the results for refinement."""
    print(
        f"\nSearching window {window['start'].date()} -> {window['stop'].date()} "
        f"at {window['step']} day cadence..."
    )
    epochs, bodies = hr.load_bodies(
        datetime_to_str(window["start"]),
        datetime_to_str(window["stop"]),
        f"{window['step']}d",
    )
    elapsed_days = np.arange(0, len(epochs)) * window["step"]
    ranked_limit = max(args.refine_top_n * 10, 50)
    stored, best_by_metric, best_cost, ranked = ls.calculate_trajectories(
        bodies,
        elapsed_days,
        args.max_revs,
        args.topk_direct,
        args.topk_ga,
        args.num_workers,
        args.max_years,
        ranked_limit=ranked_limit,
        capture_orbit=args.capture_orbit,
    )
    return {
        "window": window,
        "epochs": epochs,
        "bodies": bodies,
        "stored": stored,
        "best_by_metric": best_by_metric,
        "best_cost": best_cost,
        "ranked": ranked,
        "candidate_summaries": summarize_ranked_trajectories(ranked, epochs, window),
    }


def serialize_best_entry(best_entry, serialize_traj_fn):
    """Serialize a best-entry bundle using the caller's trajectory serializer."""
    if best_entry is None:
        return None
    return {
        "window": serialize_window(best_entry["window"]),
        "trajectory": serialize_traj_fn(best_entry["traj"], best_entry["epochs"]),
    }


def pick_better_best_entry(current_entry, candidate_entry, key):
    """Compare two stored best-entry bundles for one ranking metric."""
    if candidate_entry is None:
        return current_entry
    if current_entry is None:
        return candidate_entry

    current_traj = current_entry["traj"]
    candidate_traj = candidate_entry["traj"]
    if key == "launch":
        is_better = candidate_traj["vinf_launch_kms"] < current_traj["vinf_launch_kms"]
    elif key == "arrival":
        is_better = candidate_traj["vinf_arrive_kms"] < current_traj["vinf_arrive_kms"]
    elif key == "mission":
        is_better = traj_mission_cost(candidate_traj) < traj_mission_cost(current_traj)
    else:
        is_better = traj_total_vinf(candidate_traj) < traj_total_vinf(current_traj)

    return candidate_entry if is_better else current_entry


def execute_adaptive_search(args, serialize_traj_fn, on_level_complete=None):
    """Run the full coarse-to-fine refinement workflow across all search levels."""
    global_start = datetime.strptime(args.start, "%Y-%m-%d")
    global_stop = datetime.strptime(args.stop, "%Y-%m-%d")
    windows = [
        {
            "start": global_start,
            "stop": global_stop,
            "step": args.step,
            "source": "coarse_global",
        }
    ]
    levels = [args.step] + args.refine_steps
    level_summaries = []
    final_results = []
    global_best = {"launch": None, "arrival": None, "total": None, "mission": None}

    for level_index, level_step in enumerate(levels):
        print(
            f"\n=== Search level {level_index + 1}/{len(levels)} "
            f"({level_step} day cadence) ==="
        )
        level_results = [run_search_window(args, window) for window in windows]

        aggregate_best = {"launch": None, "arrival": None, "total": None, "mission": None}
        aggregate_candidates = []
        for result in level_results:
            aggregate_candidates.extend(result["candidate_summaries"])
            aggregate_best["launch"] = pick_better_best(
                aggregate_best["launch"],
                result["best_by_metric"]["launch"],
                result["epochs"],
                result["bodies"],
                result["window"],
                result["stored"],
                "launch",
            )
            aggregate_best["arrival"] = pick_better_best(
                aggregate_best["arrival"],
                result["best_by_metric"]["arrival"],
                result["epochs"],
                result["bodies"],
                result["window"],
                result["stored"],
                "arrival",
            )
            aggregate_best["total"] = pick_better_best(
                aggregate_best["total"],
                result["best_by_metric"]["total"],
                result["epochs"],
                result["bodies"],
                result["window"],
                result["stored"],
                "total",
            )
            aggregate_best["mission"] = pick_better_best(
                aggregate_best["mission"],
                result["best_by_metric"]["mission"],
                result["epochs"],
                result["bodies"],
                result["window"],
                result["stored"],
                "mission",
            )

        level_summaries.append(
            {
                "level": level_index + 1,
                "step_days": level_step,
                "windows": [serialize_window(window) for window in windows],
                "best_total": serialize_best_entry(
                    aggregate_best["total"], serialize_traj_fn
                ),
                "best_mission": serialize_best_entry(
                    aggregate_best["mission"], serialize_traj_fn
                ),
            }
        )

        for key in global_best:
            global_best[key] = pick_better_best_entry(
                global_best[key], aggregate_best[key], key
            )

        if on_level_complete is not None:
            on_level_complete(level_index, level_results, aggregate_best)

        if level_index == len(levels) - 1:
            final_results = level_results
            final_best = global_best
            break

        next_step = levels[level_index + 1]
        windows = select_refinement_windows(
            aggregate_candidates,
            next_step,
            args.refine_top_n,
            args.refine_pad_scale,
            global_start,
            global_stop,
        )
        if not windows:
            print("No viable refinement corridors found; stopping at current level.")
            final_results = level_results
            final_best = global_best
            break

    return final_results, final_best, level_summaries


def make_window_info(best_total_entry):
    """Build the annotation bundle used by the porkchop plot overlays."""
    return {
        "best_launch": best_total_entry["epochs"][best_total_entry["traj"]["indices"][0]],
        "best_arrival": best_total_entry["epochs"][best_total_entry["traj"]["indices"][-1]],
        "window_start": best_total_entry["window"]["start"],
        "window_end": best_total_entry["window"]["stop"],
    }
