"""Planet-centered Jupiter arrival and capture visualization helpers."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

import constants as c
from visualization.plots import PLANET_COLORS


def _to_rj(values_km):
    """Convert one position or track from kilometers to Jupiter radii."""
    return np.asarray(values_km, dtype=float) / c.R_JUPITER


def _unit_vector(vector_xy):
    """Return a normalized 2D direction, or zeros if the input is degenerate."""
    vector_xy = np.asarray(vector_xy, dtype=float)
    norm = np.linalg.norm(vector_xy)
    if norm <= 0.0:
        return np.zeros(2, dtype=float)
    return vector_xy / norm


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
        x_hat = np.array([0.0, 1.0], dtype=float) if norm <= 0.0 else direction / norm
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


def _velocity_arrow_scale(limit_rj, arrival_analysis):
    """Choose a readable arrow scale for local speed vectors."""
    speeds = np.array(
        [
            float(arrival_analysis["arrival_vinf_kms"]),
            float(arrival_analysis["hyperbolic_periapsis_speed_kms"]),
            float(arrival_analysis["capture_periapsis_speed_kms"]),
        ],
        dtype=float,
    )
    max_speed = float(np.max(speeds))
    if max_speed <= 0.0:
        return 0.0
    return 0.18 * limit_rj / max_speed


def _clip_point_to_frame(point_xy, limit_rj, margin_frac=0.08):
    """Keep annotation anchors inside the visible frame with a fixed margin."""
    margin_rj = margin_frac * limit_rj
    lower = -limit_rj + margin_rj
    upper = limit_rj - margin_rj
    point_xy = np.asarray(point_xy, dtype=float)
    return np.array(
        [
            np.clip(point_xy[0], lower, upper),
            np.clip(point_xy[1], lower, upper),
        ],
        dtype=float,
    )


def _text_alignment_for_direction(direction_xy):
    """Choose text alignment based on the direction from the annotated geometry."""
    direction_xy = _unit_vector(direction_xy)
    ha = "left" if direction_xy[0] >= 0.0 else "right"
    va = "bottom" if direction_xy[1] >= 0.0 else "top"
    return ha, va


def _label_position(
    anchor_xy, direction_xy, limit_rj, base_offset_frac=0.1, tangent_bias_xy=None
):
    """Place a label away from the anchor using a preferred quadrant and frame clipping."""
    direction_xy = _unit_vector(direction_xy)
    if np.linalg.norm(direction_xy) <= 0.0:
        direction_xy = np.array([1.0, 1.0], dtype=float) / np.sqrt(2.0)

    offset_xy = base_offset_frac * limit_rj * direction_xy
    if tangent_bias_xy is not None:
        offset_xy = offset_xy + 0.04 * limit_rj * _unit_vector(tangent_bias_xy)

    label_xy = _clip_point_to_frame(anchor_xy + offset_xy, limit_rj)
    ha, va = _text_alignment_for_direction(direction_xy)
    return label_xy, ha, va


def _add_capture_annotations(ax, arrival_analysis, tracks_rj, limit_rj):
    """Overlay the main arrival and JOI geometry cues on a capture plot."""
    periapsis_xy = np.asarray(tracks_rj["capture_track"][0], dtype=float)
    arrival_start_xy = np.asarray(tracks_rj["hyperbola_track"][0], dtype=float)
    rp_rj = float(arrival_analysis["capture_orbit"]["periapsis_radius_rj"])
    v_inf_kms = float(arrival_analysis["arrival_vinf_kms"])
    v_peri_hyp_kms = float(arrival_analysis["hyperbolic_periapsis_speed_kms"])
    v_peri_cap_kms = float(arrival_analysis["capture_periapsis_speed_kms"])
    joi_delta_v_kms = float(arrival_analysis["joi_delta_v_kms"])

    approach_hat = _unit_vector(
        tracks_rj["hyperbola_track"][1] - tracks_rj["hyperbola_track"][0]
    )
    tangent_hat = _unit_vector(
        tracks_rj["capture_track"][1] - tracks_rj["capture_track"][0]
    )
    if np.linalg.norm(approach_hat) <= 0.0:
        approach_hat = _unit_vector(periapsis_xy - arrival_start_xy)
    if np.linalg.norm(tangent_hat) <= 0.0:
        tangent_hat = _unit_vector(
            np.array([-periapsis_xy[1], periapsis_xy[0]], dtype=float)
        )
    normal_hat = _unit_vector(np.array([-tangent_hat[1], tangent_hat[0]], dtype=float))
    radial_normal_hat = _unit_vector(
        np.array([-periapsis_xy[1], periapsis_xy[0]], dtype=float)
    )

    arrow_scale = _velocity_arrow_scale(limit_rj, arrival_analysis)

    ax.plot(
        [0.0, periapsis_xy[0]],
        [0.0, periapsis_xy[1]],
        color="gray",
        linestyle=":",
        linewidth=1.2,
    )
    radius_label_xy, radius_ha, radius_va = _label_position(
        0.42 * periapsis_xy,
        radial_normal_hat + 0.35 * _unit_vector(periapsis_xy),
        limit_rj,
        base_offset_frac=0.12,
    )
    ax.text(
        radius_label_xy[0],
        radius_label_xy[1],
        f"r_periapse = {rp_rj:.2f} Rj",
        color="dimgray",
        fontsize=9,
        ha=radius_ha,
        va=radius_va,
        bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="none", alpha=0.9),
    )

    vinf_length_rj = max(0.12 * limit_rj, v_inf_kms * arrow_scale)
    vinf_tip_xy = arrival_start_xy + approach_hat * vinf_length_rj
    ax.annotate(
        "",
        xy=vinf_tip_xy,
        xytext=arrival_start_xy,
        arrowprops=dict(arrowstyle="->", color="black", linewidth=1.8),
    )
    vinf_label_xy, vinf_ha, vinf_va = _label_position(
        vinf_tip_xy,
        normal_hat + 0.55 * approach_hat,
        limit_rj,
        base_offset_frac=0.12,
        tangent_bias_xy=approach_hat,
    )
    ax.text(
        vinf_label_xy[0],
        vinf_label_xy[1],
        f"v_inf = {v_inf_kms:.2f} km/s",
        color="black",
        fontsize=9,
        ha=vinf_ha,
        va=vinf_va,
        bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="none", alpha=0.9),
    )

    hyp_tip_xy = periapsis_xy + tangent_hat * (v_peri_hyp_kms * arrow_scale)
    cap_tip_xy = periapsis_xy + tangent_hat * (v_peri_cap_kms * arrow_scale)

    ax.annotate(
        "",
        xy=hyp_tip_xy,
        xytext=periapsis_xy,
        arrowprops=dict(arrowstyle="->", color="black", linewidth=1.8),
    )
    ax.annotate(
        "",
        xy=cap_tip_xy,
        xytext=periapsis_xy,
        arrowprops=dict(arrowstyle="->", color="teal", linewidth=1.8),
    )
    ax.annotate(
        "",
        xy=hyp_tip_xy,
        xytext=cap_tip_xy,
        arrowprops=dict(arrowstyle="<->", color="crimson", linewidth=1.6),
    )

    pre_label_xy, pre_ha, pre_va = _label_position(
        hyp_tip_xy,
        normal_hat + 0.45 * tangent_hat,
        limit_rj,
        base_offset_frac=0.14,
        tangent_bias_xy=tangent_hat,
    )
    post_label_xy, post_ha, post_va = _label_position(
        cap_tip_xy,
        -normal_hat + 0.6 * tangent_hat,
        limit_rj,
        base_offset_frac=0.16,
        tangent_bias_xy=tangent_hat,
    )
    dv_label_xy, dv_ha, dv_va = _label_position(
        0.5 * (hyp_tip_xy + cap_tip_xy),
        -normal_hat + 0.25 * tangent_hat,
        limit_rj,
        base_offset_frac=0.2,
    )
    ax.text(
        dv_label_xy[0],
        dv_label_xy[1],
        f"JOI delta-v = {joi_delta_v_kms:.2f} km/s",
        color="crimson",
        fontsize=9,
        ha=dv_ha,
        va=dv_va,
        bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="none", alpha=0.92),
    )


def _build_animation_schedule(t_hyp_rel_s, t_cap_s):
    """Create explicit animation phases so the JOI event has time to read."""
    schedule = []

    for time_s in np.linspace(t_hyp_rel_s[0], 0.0, 120, endpoint=True):
        schedule.append(("approach", float(time_s)))
    for _ in range(24):
        schedule.append(("periapsis_hold", 0.0))
    for _ in range(24):
        schedule.append(("burn", 0.0))
    for time_s in np.linspace(0.0, t_cap_s[-1], 192, endpoint=True):
        schedule.append(("capture", float(time_s)))

    return schedule


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
    tracks_rj = {
        "t_hyp_s": tracks["t_hyp_s"],
        "hyperbola_track": _to_rj(tracks["hyperbola_track"]),
        "t_cap_s": tracks["t_cap_s"],
        "capture_track": _to_rj(tracks["capture_track"]),
    }
    limit_rj = 1.2 * (_capture_axis_limit_km(tracks) / c.R_JUPITER)
    periapsis_xy = tracks_rj["capture_track"][0]

    plot_outfile = output_dir / f"jupiter_capture_plot_{tag}.png"
    animation_outfile = output_dir / f"jupiter_capture_animation_{tag}.mp4"

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    jupiter = plt.Circle(
        (0.0, 0.0), 1.0, color=PLANET_COLORS["Jupiter"], label="Jupiter"
    )
    ax.add_patch(jupiter)
    ax.plot(
        tracks_rj["hyperbola_track"][:, 0],
        tracks_rj["hyperbola_track"][:, 1],
        color="black",
        linestyle="--",
        label="Inbound hyperbola",
    )
    ax.plot(
        tracks_rj["capture_track"][:, 0],
        tracks_rj["capture_track"][:, 1],
        color="teal",
        label="Captured orbit",
    )
    ax.scatter(
        tracks_rj["hyperbola_track"][0, 0],
        tracks_rj["hyperbola_track"][0, 1],
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
        tracks_rj["capture_track"][-1, 0],
        tracks_rj["capture_track"][-1, 1],
        color="teal",
        marker=">",
        s=70,
        label="Spacecraft",
    )
    _add_capture_annotations(ax, arrival_analysis, tracks_rj, limit_rj)

    ax.set_aspect("equal")
    ax.set_xlim(-limit_rj, limit_rj)
    ax.set_ylim(-limit_rj, limit_rj)
    ax.set_xlabel("Jupiter-centered x (Rj)")
    ax.set_ylabel("Jupiter-centered y (Rj)")
    ax.set_title("Top-down Jupiter arrival and JOI geometry")

    sun_marker_xy = _sun_marker_position(sun_direction_xy_km, limit_rj)
    if sun_marker_xy is not None:
        ax.scatter(
            sun_marker_xy[0],
            sun_marker_xy[1],
            color="gold",
            edgecolors="black",
            s=120,
            zorder=6,
        )
        ax.text(
            sun_marker_xy[0] - (0.07 * limit_rj),
            sun_marker_xy[1],
            "Sun (not to scale)",
            color="goldenrod",
            fontsize=9,
            va="center",
            ha="left" if sun_marker_xy[0] >= 0.0 else "right",
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.88),
        )

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.96, 1.0),
        borderaxespad=0.0,
        fontsize=8,
        framealpha=0.95,
    )
    fig.tight_layout()
    fig.savefig(plot_outfile, dpi=200, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)

    _animate_jupiter_capture(
        tracks, arrival_analysis, animation_outfile, limit_rj, sun_direction_xy_km
    )
    return plot_outfile, animation_outfile


def _animate_jupiter_capture(
    tracks, arrival_analysis, outfile, limit_rj, sun_direction_xy_km=None
):
    """Animate the inbound hyperbola, JOI event, and subsequent captured orbits."""
    t_hyp_rel_s = tracks["t_hyp_s"] - tracks["t_hyp_s"][-1]
    t_cap_s = tracks["t_cap_s"]
    hyperbola_track = _to_rj(tracks["hyperbola_track"])
    capture_track = _to_rj(tracks["capture_track"])
    periapsis_xy = capture_track[0]
    frame_schedule = _build_animation_schedule(t_hyp_rel_s, t_cap_s)

    fig, ax = plt.subplots(figsize=(9.5, 9.5))
    ax.set_aspect("equal")
    ax.set_xlim(-limit_rj, limit_rj)
    ax.set_ylim(-limit_rj, limit_rj)
    ax.set_xlabel("Jupiter-centered x (Rj)")
    ax.set_ylabel("Jupiter-centered y (Rj)")
    ax.set_title("Top-down Jupiter arrival and JOI animation")

    jupiter = plt.Circle((0.0, 0.0), 1.0, color=PLANET_COLORS["Jupiter"], alpha=0.8)
    ax.add_patch(jupiter)
    _add_capture_annotations(
        ax,
        arrival_analysis,
        {
            "hyperbola_track": hyperbola_track,
            "capture_track": capture_track,
        },
        limit_rj,
    )

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
    sun_marker_xy = _sun_marker_position(sun_direction_xy_km, limit_rj)
    if sun_marker_xy is not None:
        (sun_pt,) = ax.plot(
            [sun_marker_xy[0]],
            [sun_marker_xy[1]],
            marker="o",
            markersize=10,
            color="gold",
            markeredgecolor="black",
        )
        sun_text = ax.text(
            sun_marker_xy[0] - (0.07 * limit_rj),
            sun_marker_xy[1],
            " Sun (not to scale)",
            color="goldenrod",
            fontsize=9,
            va="center",
            ha="left" if sun_marker_xy[0] >= 0.0 else "right",
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.88),
        )

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.96, 1.0),
        borderaxespad=0.0,
        fontsize=8,
        framealpha=0.95,
    )

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
        phase, frame_time_s = frame_schedule[frame_number]
        burn_pt.set_data([], [])

        if phase == "approach":
            hyp_indices = np.where(t_hyp_rel_s <= frame_time_s)[0]
            last_idx = int(hyp_indices[-1]) if hyp_indices.size else 0
            sc_xy = _interp_track_point(frame_time_s, t_hyp_rel_s, hyperbola_track)
            hyper_line.set_data(
                hyperbola_track[: last_idx + 1, 0], hyperbola_track[: last_idx + 1, 1]
            )
            capture_line.set_data([], [])
            sc_pt.set_data([sc_xy[0]], [sc_xy[1]])
            event_text.set_text(
                "Phase: Inbound approach\n"
                f"Jupiter-relative v_inf = {arrival_analysis['arrival_vinf_kms']:.2f} km/s"
            )
        elif phase == "periapsis_hold":
            hyper_line.set_data(hyperbola_track[:, 0], hyperbola_track[:, 1])
            capture_line.set_data([], [])
            sc_pt.set_data([periapsis_xy[0]], [periapsis_xy[1]])
            burn_pt.set_data([periapsis_xy[0]], [periapsis_xy[1]])
            event_text.set_text(
                "Phase: Periapsis arrival\n"
                f"Periapsis radius = {arrival_analysis['capture_orbit']['periapsis_radius_rj']:.2f} Rj"
            )
        elif phase == "burn":
            hyper_line.set_data(hyperbola_track[:, 0], hyperbola_track[:, 1])
            capture_line.set_data([], [])
            sc_pt.set_data([periapsis_xy[0]], [periapsis_xy[1]])
            if frame_number % 6 < 3:
                burn_pt.set_data([periapsis_xy[0]], [periapsis_xy[1]])
            event_text.set_text(
                "Phase: JOI burn\n"
                f"Delta-v = {arrival_analysis['joi_delta_v_kms']:.2f} km/s"
            )
        else:
            cap_indices = np.where(t_cap_s <= frame_time_s)[0]
            last_idx = int(cap_indices[-1]) if cap_indices.size else 0
            sc_xy = _interp_track_point(frame_time_s, t_cap_s, capture_track)
            hyper_line.set_data(hyperbola_track[:, 0], hyperbola_track[:, 1])
            capture_line.set_data(
                capture_track[: last_idx + 1, 0], capture_track[: last_idx + 1, 1]
            )
            sc_pt.set_data([sc_xy[0]], [sc_xy[1]])
            burn_pt.set_data([periapsis_xy[0]], [periapsis_xy[1]])
            event_text.set_text(
                "Phase: Captured orbit\n"
                f"Post-JOI periapsis speed = {arrival_analysis['capture_periapsis_speed_kms']:.2f} km/s"
            )

        artists = [hyper_line, capture_line, sc_pt, burn_pt, event_text]
        if sun_pt is not None:
            artists.append(sun_pt)
        if sun_text is not None:
            artists.append(sun_text)
        return tuple(artists)

    ani = FuncAnimation(
        fig, update, frames=len(frame_schedule), init_func=init, blit=True, interval=35
    )
    ani.save(outfile, writer="ffmpeg", fps=20)
    plt.close(fig)
