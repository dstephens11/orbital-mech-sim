"""Helpers for writing and reloading saved trajectory snapshots."""

import json
from pathlib import Path

import numpy as np

SNAPSHOT_BEST_LABELS = ("total", "mission", "launch", "arrival")


def _datetime_to_str(value):
    """Format a datetime-like value as the YYYY-MM-DD string used in snapshots."""
    return value.strftime("%Y-%m-%d")


def serialize_window(window):
    """Convert a search-window definition into JSON-friendly data."""
    return {
        "start": _datetime_to_str(window["start"]),
        "stop": _datetime_to_str(window["stop"]),
        "step_days": window["step"],
        "source": window.get("source", "manual"),
    }


def _serialize_leg(leg):
    """Convert one Lambert leg into JSON-friendly data."""
    return {
        "from": leg["from"],
        "to": leg["to"],
        "i_dep": leg["i_dep"],
        "i_arr": leg["i_arr"],
        "tof_days": leg["tof_days"],
        "v1_aud": np.asarray(leg["v1_aud"], dtype=float).tolist(),
        "v2_aud": np.asarray(leg["v2_aud"], dtype=float).tolist(),
        "vinf_dep_kms": leg["vinf_dep_kms"],
        "vinf_arr_kms": leg["vinf_arr_kms"],
    }


def serialize_snapshot_traj(traj, epochs, serialize_traj_fn):
    """Convert a stored trajectory into a reusable snapshot record."""
    snapshot = serialize_traj_fn(traj, epochs)
    snapshot["lambert_legs"] = [_serialize_leg(leg) for leg in traj["lambert_legs"]]
    if "flyby" in traj:
        snapshot["flyby"] = traj["flyby"]
    return snapshot


def _snapshot_ephemeris_arrays(entry):
    """Convert one best-entry ephemeris bundle into NPZ-ready arrays."""
    arrays = {
        "epochs_iso": np.array(
            [epoch.isoformat() for epoch in entry["epochs"]], dtype="<U32"
        )
    }
    for body_name, body in entry["bodies"].items():
        body_slug = body_name.lower()
        arrays[f"{body_slug}_pos_au"] = np.asarray(body["pos"], dtype=float)
        arrays[f"{body_slug}_vel_au_per_day"] = np.asarray(body["vel"], dtype=float)
    return arrays


def build_snapshot_payload(best_entries, summary, serialize_traj_fn):
    """Build the reusable JSON payload saved alongside each search run."""
    payload = {
        "config": summary["config"],
        "capture_model": summary["capture_model"],
        "search_levels": summary["search_levels"],
    }
    for label in SNAPSHOT_BEST_LABELS:
        entry = best_entries[label]
        payload[f"best_{label}"] = {
            "window": serialize_window(entry["window"]),
            "trajectory": serialize_snapshot_traj(
                entry["traj"], entry["epochs"], serialize_traj_fn
            ),
        }
    return payload


def write_trajectory_snapshot(output_dir, best_entries, summary, serialize_traj_fn):
    """Write reusable trajectory and ephemeris snapshots for downstream analysis."""
    snapshot_payload = build_snapshot_payload(best_entries, summary, serialize_traj_fn)
    output_dir = Path(output_dir)
    (output_dir / "trajectory_ephemeris_snapshot.json").write_text(
        json.dumps(snapshot_payload, indent=2)
    )

    ephemeris_arrays = {}
    for label in SNAPSHOT_BEST_LABELS:
        entry = best_entries[label]
        for name, values in _snapshot_ephemeris_arrays(entry).items():
            ephemeris_arrays[f"{label}_{name}"] = values
    np.savez_compressed(output_dir / "trajectory_ephemeris_snapshot.npz", **ephemeris_arrays)


def load_snapshot(snapshot_path):
    """Load a saved trajectory snapshot JSON file."""
    return json.loads(Path(snapshot_path).read_text())


def load_snapshot_npz(snapshot_npz_path):
    """Load a saved trajectory snapshot NPZ file."""
    return np.load(Path(snapshot_npz_path))


def get_snapshot_best(snapshot, best_label):
    """Return one saved best-trajectory bundle from a snapshot payload."""
    key = f"best_{best_label}"
    if key not in snapshot:
        available = ", ".join(
            key.removeprefix("best_") for key in snapshot.keys() if key.startswith("best_")
        )
        raise KeyError(
            f"Snapshot does not contain '{key}'. Available saved trajectories: {available or 'none'}."
        )
    return snapshot[key]
