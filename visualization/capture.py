"""Planet-centered Jupiter arrival and capture visualization helpers."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

import constants as c
from visualization.plots import PLANET_COLORS


def _rotate_track(track_xy, angle_rad):
    """Rotate a 2D track by a fixed angle."""
    rotation = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)],
        ],
        dtype=float,
    )
    return np.asarray(track_xy, dtype=float) @ rotation.T


def _hyperbolic_arrival_track(
    arrival_analysis, branch_sign=-1.0, n_samples=240, start_radius_km=None
):
    """Build a representative inbound hyperbolic arrival arc ending at periapsis."""
    capture_orbit = arrival_analysis["capture_orbit"]
    v_inf_kms = float(arrival_analysis["arrival_vinf_kms"])
    rp_km = float(capture_orbit["periapsis_radius_km"])
    ra_km = float(capture_orbit["apoapsis_radius_km"])

    if start_radius_km is None:
        start_radius_km = max(1.2 * ra_km, 10.0 * rp_km)

    a_abs_km = c.MU_JUPITER / (v_inf_kms**2)
    ecc = 1.0 + (rp_km * v_inf_kms**2) / c.MU_JUPITER
    p_km = rp_km * (1.0 + ecc)

    cos_f_start = np.clip((p_km / start_radius_km - 1.0) / ecc, -1.0, 1.0)
    f_start_abs = float(np.arccos(cos_f_start))
    true_anomaly = np.linspace(branch_sign * f_start_abs, 0.0, n_samples)
    radius_km = p_km / (1.0 + ecc * np.cos(true_anomaly))

    x_km = radius_km * np.cos(true_anomaly)
    y_km = radius_km * np.sin(true_anomaly)

    scale = np.sqrt((ecc - 1.0) / (ecc + 1.0))
    hyperbolic_anomaly = 2.0 * np.arctanh(scale * np.tan(true_anomaly / 2.0))
    mean_anomaly = ecc * np.sinh(hyperbolic_anomaly) - hyperbolic_anomaly
    mean_motion = np.sqrt(c.MU_JUPITER / (a_abs_km**3))
    time_s = mean_anomaly / mean_motion
    time_s = time_s - time_s[0]

    return time_s, np.column_stack((x_km, y_km))


def _solve_kepler_elliptic(mean_anomaly, eccentricity, max_iter=25):
    """Solve Kepler's equation for an elliptical orbit."""
    mean_anomaly = np.asarray(mean_anomaly, dtype=float)
    mean_wrapped = np.mod(mean_anomaly, 2.0 * np.pi)
    revolution_offset = mean_anomaly - mean_wrapped

    eccentric_anomaly = mean_wrapped.copy()
    if eccentricity > 0.8:
        eccentric_anomaly = np.where(
            mean_wrapped < np.pi,
            mean_wrapped + 0.85 * eccentricity,
            mean_wrapped - 0.85 * eccentricity,
        )

    for _ in range(max_iter):
        residual = (
            eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly) - mean_wrapped
        )
        derivative = 1.0 - eccentricity * np.cos(eccentric_anomaly)
        step = residual / derivative
        eccentric_anomaly -= step
        if np.max(np.abs(step)) < 1e-12:
            break

    return eccentric_anomaly + revolution_offset


def _captured_orbit_track(arrival_analysis, n_orbits=2, samples_per_orbit=240):
    """Build the post-burn captured elliptical orbit track."""
    capture_orbit = arrival_analysis["capture_orbit"]
    rp_km = float(capture_orbit["periapsis_radius_km"])
    ra_km = float(capture_orbit["apoapsis_radius_km"])

    semi_major_axis_km = 0.5 * (rp_km + ra_km)
    eccentricity = (ra_km - rp_km) / (ra_km + rp_km)
    total_samples = max(2, int(n_orbits * samples_per_orbit))
    mean_anomaly = np.linspace(0.0, 2.0 * np.pi * n_orbits, total_samples)
    eccentric_anomaly = _solve_kepler_elliptic(mean_anomaly, eccentricity)
    semi_minor_axis_km = semi_major_axis_km * np.sqrt(1.0 - eccentricity**2)
    x_km = semi_major_axis_km * (np.cos(eccentric_anomaly) - eccentricity)
    y_km = semi_minor_axis_km * np.sin(eccentric_anomaly)

    mean_motion = np.sqrt(c.MU_JUPITER / (semi_major_axis_km**3))
    time_s = mean_anomaly / mean_motion
    return time_s, np.column_stack((x_km, y_km))


def _capture_rotation_rad(sun_direction_xy_km=None):
    """Choose the display rotation for the approximate local capture geometry."""
    if sun_direction_xy_km is None:
        return 0.5 * np.pi

    direction = np.asarray(sun_direction_xy_km, dtype=float)
    if np.linalg.norm(direction) <= 0.0:
        return 0.5 * np.pi

    # By convention, the approximate captured ellipse is drawn with periapsis
    # on the Sunward side of Jupiter and apoapsis on the anti-Sunward side.
    return float(np.arctan2(direction[1], direction[0]))


def _local_sun_frame(sun_direction_xy_km=None):
    """Return a 2D local frame with +x toward the Sun and +y completing the plane."""
    if sun_direction_xy_km is None:
        x_hat = np.array([0.0, 1.0], dtype=float)
    else:
        direction = np.asarray(sun_direction_xy_km, dtype=float)
        norm = np.linalg.norm(direction)
        x_hat = (
            np.array([0.0, 1.0], dtype=float)
            if norm <= 0.0
            else direction / norm
        )
    y_hat = np.array([-x_hat[1], x_hat[0]], dtype=float)
    return x_hat, y_hat


def _incoming_branch_sign(arrival_analysis, sun_direction_xy_km=None):
    """Choose the displayed inbound hyperbolic branch from the saved arrival v_inf vector."""
    arrival_vinf_xy = np.asarray(
        arrival_analysis["arrival_vinf_vec_kms"][:2], dtype=float
    )
    vinf_norm = np.linalg.norm(arrival_vinf_xy)
    if vinf_norm <= 0.0:
        return -1.0

    x_hat, y_hat = _local_sun_frame(sun_direction_xy_km)
    local_vinf_y = float(np.dot(arrival_vinf_xy, y_hat))
    # Positive incoming transverse velocity means the approach should come
    # from the -y side of the local Sun-referenced frame, and vice versa.
    return -1.0 if local_vinf_y >= 0.0 else 1.0


def _capture_tracks(arrival_analysis, n_orbits=2, sun_direction_xy_km=None):
    """Build representative inbound and post-burn tracks for Jupiter capture plots."""
    branch_sign = _incoming_branch_sign(arrival_analysis, sun_direction_xy_km)
    t_hyp_s, hyperbola_track = _hyperbolic_arrival_track(
        arrival_analysis, branch_sign=branch_sign
    )
    t_cap_s, capture_track = _captured_orbit_track(arrival_analysis, n_orbits=n_orbits)
    rotation_rad = _capture_rotation_rad(sun_direction_xy_km)
    hyperbola_track = _rotate_track(hyperbola_track, rotation_rad)
    capture_track = _rotate_track(capture_track, rotation_rad)
    return {
        "t_hyp_s": t_hyp_s,
        "hyperbola_track": hyperbola_track,
        "t_cap_s": t_cap_s,
        "capture_track": capture_track,
    }


def _capture_axis_limit_km(tracks):
    """Choose a symmetric plot radius that fits arrival and captured-orbit motion."""
    stacked = np.vstack((tracks["hyperbola_track"], tracks["capture_track"]))
    return 1.08 * float(np.max(np.linalg.norm(stacked, axis=1)))


def _interp_track_point(time_s, track_times_s, track_xy):
    """Interpolate one 2D position along a track at a given physical time."""
    return np.array(
        [
            np.interp(time_s, track_times_s, track_xy[:, 0]),
            np.interp(time_s, track_times_s, track_xy[:, 1]),
        ],
        dtype=float,
    )


def _sun_marker_position(sun_direction_xy_km, limit_km):
    """Place a readable Sun marker near the frame edge along the Sun direction."""
    direction = np.asarray(sun_direction_xy_km, dtype=float)
    norm = np.linalg.norm(direction)
    if norm <= 0.0:
        return None
    return direction / norm * (0.88 * limit_km)


def plot_jupiter_capture(
    arrival_analysis,
    tag="mission",
    output_dir: Path = Path("."),
    n_orbits=2,
    sun_direction_xy_km=None,
):
    """Save a representative Jupiter-centered capture plot and animation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tracks = _capture_tracks(
        arrival_analysis, n_orbits=n_orbits, sun_direction_xy_km=sun_direction_xy_km
    )
    limit_km = _capture_axis_limit_km(tracks)
    periapsis_xy = tracks["capture_track"][0]

    plot_outfile = output_dir / f"jupiter_capture_plot_{tag}.png"
    animation_outfile = output_dir / f"jupiter_capture_animation_{tag}.mp4"

    fig, ax = plt.subplots(figsize=(7, 7))
    jupiter = plt.Circle(
        (0.0, 0.0), c.R_JUPITER, color=PLANET_COLORS["Jupiter"], label="Jupiter"
    )
    ax.add_patch(jupiter)
    ax.plot(
        tracks["hyperbola_track"][:, 0],
        tracks["hyperbola_track"][:, 1],
        color="black",
        linestyle="--",
        label="Inbound hyperbola",
    )
    ax.plot(
        tracks["capture_track"][:, 0],
        tracks["capture_track"][:, 1],
        color="teal",
        label="Captured orbit",
    )
    ax.scatter(
        tracks["hyperbola_track"][0, 0],
        tracks["hyperbola_track"][0, 1],
        color="black",
        s=35,
        label="Arrival start",
    )
    ax.scatter(
        periapsis_xy[0],
        periapsis_xy[1],
        color="red",
        marker="*",
        s=120,
        label="JOI burn",
    )
    ax.scatter(
        tracks["capture_track"][-1, 0],
        tracks["capture_track"][-1, 1],
        color="teal",
        marker=">",
        s=70,
        label="Spacecraft",
    )

    ax.set_aspect("equal")
    ax.set_xlim(-limit_km, limit_km)
    ax.set_ylim(-limit_km, limit_km)
    ax.set_xlabel("Jupiter-centered x (km)")
    ax.set_ylabel("Jupiter-centered y (km)")
    ax.set_title("Top-down Jupiter arrival and capture view")

    sun_marker_xy = _sun_marker_position(sun_direction_xy_km, limit_km)
    if sun_marker_xy is not None:
        ax.scatter(
            sun_marker_xy[0],
            sun_marker_xy[1],
            color="gold",
            edgecolors="black",
            s=120,
            zorder=6,
            label="Sun direction",
        )
        ax.text(
            sun_marker_xy[0] - (0.04 * limit_km),
            sun_marker_xy[1],
            "Sun",
            color="goldenrod",
            fontsize=9,
            va="center",
            ha="left" if sun_marker_xy[0] >= 0.0 else "right",
        )

    ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(plot_outfile, dpi=200, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)

    _animate_jupiter_capture(tracks, animation_outfile, limit_km, sun_direction_xy_km)
    return plot_outfile, animation_outfile


def _animate_jupiter_capture(tracks, outfile, limit_km, sun_direction_xy_km=None):
    """Animate the inbound hyperbola, JOI event, and subsequent captured orbits."""
    t_hyp_rel_s = tracks["t_hyp_s"] - tracks["t_hyp_s"][-1]
    t_cap_s = tracks["t_cap_s"]
    hyperbola_track = tracks["hyperbola_track"]
    capture_track = tracks["capture_track"]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.set_xlim(-limit_km, limit_km)
    ax.set_ylim(-limit_km, limit_km)
    ax.set_xlabel("Jupiter-centered x (km)")
    ax.set_ylabel("Jupiter-centered y (km)")
    ax.set_title("Top-down Jupiter arrival and JOI animation")

    jupiter = plt.Circle(
        (0.0, 0.0), c.R_JUPITER, color=PLANET_COLORS["Jupiter"], alpha=0.8
    )
    ax.add_patch(jupiter)

    (hyper_line,) = ax.plot(
        [], [], color="black", linestyle="--", lw=1.5, label="Inbound hyperbola"
    )
    (capture_line,) = ax.plot([], [], color="teal", lw=1.5, label="Captured orbit")
    (sc_pt,) = ax.plot(
        [], [], marker=">", color="black", markersize=8, label="Spacecraft"
    )
    (burn_pt,) = ax.plot(
        [], [], marker="*", color="red", markersize=12, label="JOI burn"
    )
    event_text = ax.text(
        0.02,
        0.97,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.9),
    )

    sun_pt = None
    sun_text = None
    sun_marker_xy = _sun_marker_position(sun_direction_xy_km, limit_km)
    if sun_marker_xy is not None:
        (sun_pt,) = ax.plot(
            [sun_marker_xy[0]],
            [sun_marker_xy[1]],
            marker="o",
            markersize=10,
            color="gold",
            markeredgecolor="black",
            label="Sun direction",
        )
        sun_text = ax.text(
            sun_marker_xy[0] - (0.04 * limit_km),
            sun_marker_xy[1],
            " Sun",
            color="goldenrod",
            fontsize=9,
            va="center",
            ha="left" if sun_marker_xy[0] >= 0.0 else "right",
        )

    ax.legend(loc="upper right", fontsize=8, framealpha=0.95)

    total_frames = 360
    frame_times_s = np.linspace(t_hyp_rel_s[0], t_cap_s[-1], total_frames)
    frame_dt_s = frame_times_s[1] - frame_times_s[0] if total_frames > 1 else 0.0
    periapsis_xy = capture_track[0]

    def init():
        """Reset all artists before the animation starts."""
        hyper_line.set_data([], [])
        capture_line.set_data([], [])
        sc_pt.set_data([], [])
        burn_pt.set_data([], [])
        event_text.set_text("")
        artists = [hyper_line, capture_line, sc_pt, burn_pt, event_text]
        if sun_pt is not None:
            artists.append(sun_pt)
        if sun_text is not None:
            artists.append(sun_text)
        return tuple(artists)

    def update(frame_number):
        """Advance the Jupiter arrival/capture animation by one frame."""
        frame_time_s = frame_times_s[frame_number]
        burn_pt.set_data([], [])

        if frame_time_s < 0.0:
            hyp_indices = np.where(t_hyp_rel_s <= frame_time_s)[0]
            last_idx = int(hyp_indices[-1]) if hyp_indices.size else 0
            sc_xy = _interp_track_point(frame_time_s, t_hyp_rel_s, hyperbola_track)
            hyper_line.set_data(hyperbola_track[: last_idx + 1, 0], hyperbola_track[: last_idx + 1, 1])
            capture_line.set_data([], [])
            sc_pt.set_data([sc_xy[0]], [sc_xy[1]])
            event_text.set_text("Inbound hyperbolic approach")
        else:
            cap_indices = np.where(t_cap_s <= frame_time_s)[0]
            last_idx = int(cap_indices[-1]) if cap_indices.size else 0
            sc_xy = _interp_track_point(frame_time_s, t_cap_s, capture_track)
            hyper_line.set_data(hyperbola_track[:, 0], hyperbola_track[:, 1])
            capture_line.set_data(capture_track[: last_idx + 1, 0], capture_track[: last_idx + 1, 1])
            sc_pt.set_data([sc_xy[0]], [sc_xy[1]])
            if abs(frame_time_s) <= max(frame_dt_s, 1e-9):
                burn_pt.set_data([periapsis_xy[0]], [periapsis_xy[1]])
                event_text.set_text("JOI burn at periapsis")
            else:
                burn_pt.set_data([periapsis_xy[0]], [periapsis_xy[1]])
                event_text.set_text("Captured orbit after JOI")

        artists = [hyper_line, capture_line, sc_pt, burn_pt, event_text]
        if sun_pt is not None:
            artists.append(sun_pt)
        if sun_text is not None:
            artists.append(sun_text)
        return tuple(artists)

    ani = FuncAnimation(
        fig, update, frames=total_frames, init_func=init, blit=True, interval=30
    )
    ani.save(outfile, writer="ffmpeg", fps=20)
    plt.close(fig)
