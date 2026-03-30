"""Porkchop, trajectory, and animation plotting helpers."""

from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from dynamics import propagation as prop

PLANET_COLORS = {
    "Venus": "gold",
    "Earth": "blue",
    "Mars": "red",
    "Jupiter": "orange",
}

PLOT_SPECS = [
    ("Direct", "direct", "Porkchop: Direct Earth→Jupiter"),
    ("1-GA", "1ga", "Porkchop: 1 Gravity Assist"),
    ("2-GA", "2ga", "Porkchop: 2 Gravity Assists"),
]


def traj_class(traj):
    """Map trajectory type strings into the three porkchop plot classes."""
    traj_type = traj.get("type", "")
    if traj_type == "Direct":
        return "Direct"
    if (" GA" in traj_type) and ("-" not in traj_type):
        return "1-GA"
    if (" GA" in traj_type) and ("-" in traj_type):
        return "2-GA"
    return "Other"


def make_porkchop(stored, epochs, class_name, plot_type="launch"):
    """Build a launch-vs-arrival cost grid for one trajectory class."""
    num_epochs = len(epochs)
    grid = np.full((num_epochs, num_epochs), np.nan, dtype=float)

    for traj in stored:
        if traj_class(traj) != class_name:
            continue

        launch_index = traj["indices"][0]
        arrival_index = traj["indices"][-1]
        if arrival_index <= launch_index:
            continue

        if plot_type == "launch":
            cost = traj["vinf_launch_kms"]
        elif plot_type == "arrival":
            cost = traj["vinf_arrive_kms"]
        elif plot_type == "total":
            cost = traj["vinf_launch_kms"] + traj["vinf_arrive_kms"]
        elif plot_type == "mission":
            cost = traj["mission_total_cost_kms"]
        else:
            raise ValueError(f"Invalid cost type: {plot_type}")

        if np.isnan(grid[arrival_index, launch_index]) or cost < grid[arrival_index, launch_index]:
            grid[arrival_index, launch_index] = cost

    return grid


def _has_porkchop_data(grid):
    """Return whether a porkchop grid contains any finite trajectory data."""
    return np.isfinite(grid).any()


def _porkchop_bounds(grid, epochs):
    """Return the finite-data bounds of a porkchop cost grid."""
    finite_rows, finite_cols = np.where(np.isfinite(grid))
    if finite_rows.size == 0 or finite_cols.size == 0:
        return epochs[0], epochs[-1], epochs[0], epochs[-1]

    launch_start = epochs[finite_cols.min()]
    launch_end = epochs[finite_cols.max()]
    arrival_start = epochs[finite_rows.min()]
    arrival_end = epochs[finite_rows.max()]
    return launch_start, launch_end, arrival_start, arrival_end


def build_plot(grid, epochs, title):
    """Create a formatted porkchop figure and return its plotting handles."""
    x = mdates.date2num(epochs)
    y = mdates.date2num(epochs)

    fig, ax = plt.subplots(figsize=(9, 7))
    X, Y = np.meshgrid(x, y)
    mesh = ax.pcolormesh(X, Y, grid, vmin=5, vmax=20, shading="auto")
    colorbar = plt.colorbar(mesh, ax=ax)

    ax.set_title(title)
    ax.set_xlabel("Launch date (Earth)")
    ax.set_ylabel("Arrival date (Jupiter)")
    ax.xaxis_date()
    ax.yaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    launch_start, launch_end, arrival_start, arrival_end = _porkchop_bounds(grid, epochs)
    ax.set_xlim(mdates.date2num(launch_start), mdates.date2num(launch_end))
    ax.set_ylim(mdates.date2num(arrival_start), mdates.date2num(arrival_end))
    fig.autofmt_xdate()

    return fig, ax, colorbar, arrival_start, arrival_end


def save_plot(fig, colorbar, outfile, plot_type="launch"):
    """Label and save a porkchop figure to disk."""
    if plot_type == "launch":
        colorbar.set_label(r"$v_{\infty,launch}$ (km/s)")
    elif plot_type == "arrival":
        colorbar.set_label(r"$v_{\infty,arrive}$ (km/s)")
    elif plot_type == "total":
        colorbar.set_label(r"$v_{\infty,launch} + v_{\infty,arrive}$ (km/s)")
    elif plot_type == "mission":
        colorbar.set_label(r"$v_{\infty,launch} + \Delta v_{JOI}$ (km/s)")
    else:
        raise ValueError(f"Invalid cost type: {plot_type}")

    fig.tight_layout(pad=1.2)
    fig.savefig(outfile, dpi=200, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def plot_porkchop(grid, epochs, title, outfile: Path, plot_type="launch"):
    """Render and save one porkchop plot."""
    fig, _, colorbar, _, _ = build_plot(grid, epochs, title)
    save_plot(fig, colorbar, outfile, plot_type)


def _closest_launch_column_value(grid, epochs, launch_date):
    """Return the minimum cost in the launch column nearest a given date."""
    try:
        launch_index = np.argmin(np.abs(np.array(epochs) - launch_date))
        column = grid[:, launch_index]
        return np.nan if np.all(np.isnan(column)) else np.nanmin(column)
    except Exception:
        return np.nan


def _boundary_launch_annotation(grid, epochs, launch_date, side):
    """Pick a finite launch column to annotate a window boundary."""
    epoch_array = np.array(epochs)
    finite_cols = np.where(np.isfinite(grid).any(axis=0))[0]
    if finite_cols.size == 0:
        return None, np.nan

    target_index = int(np.argmin(np.abs(epoch_array - launch_date)))
    if side == "start":
        candidate_cols = finite_cols[finite_cols >= target_index]
        if candidate_cols.size == 0:
            candidate_cols = finite_cols
        launch_index = int(candidate_cols[0])
    elif side == "end":
        candidate_cols = finite_cols[finite_cols <= target_index]
        if candidate_cols.size == 0:
            candidate_cols = finite_cols
        launch_index = int(candidate_cols[-1])
    else:
        raise ValueError(f"Invalid boundary side: {side}")

    column = grid[:, launch_index]
    return epochs[launch_index], float(np.nanmin(column))


def plot_annotated_porkchop(
    grid,
    epochs,
    title,
    window_info: dict[str, datetime],
    outfile: Path,
    plot_type="launch",
):
    """Overlay the best point and the final refinement window on a porkchop plot."""
    fig, ax, colorbar, arrival_start, arrival_end = build_plot(grid, epochs, title)

    best_launch = window_info["best_launch"]
    best_arrival = window_info["best_arrival"]
    window_start = window_info["window_start"]
    window_end = window_info["window_end"]

    best_x = mdates.date2num(best_launch)
    best_y = mdates.date2num(best_arrival)
    win_x0 = mdates.date2num(window_start)
    win_x1 = mdates.date2num(window_end)

    vinf_best = _closest_launch_column_value(grid, epochs, best_launch)
    start_launch_date, vinf_start = _boundary_launch_annotation(
        grid, epochs, window_start, "start"
    )
    end_launch_date, vinf_end = _boundary_launch_annotation(
        grid, epochs, window_end, "end"
    )

    ax.plot(best_x, best_y, marker="*", markersize=12, markeredgecolor="black", markerfacecolor="red", zorder=5)
    ax.annotate(
        f"Best\n$v_\\infty$={vinf_best:.2f} km/s",
        xy=(best_x, best_y),
        xytext=(15, 15),
        textcoords="offset points",
        fontsize=10,
        bbox=dict(boxstyle="round", fc="white", ec="black"),
        arrowprops=dict(arrowstyle="->"),
        annotation_clip=False,
    )

    rect = patches.Rectangle(
        (win_x0, mdates.date2num(arrival_start)),
        win_x1 - win_x0,
        mdates.date2num(arrival_end) - mdates.date2num(arrival_start),
        linewidth=2,
        edgecolor="white",
        facecolor="none",
        linestyle="--",
    )
    ax.add_patch(rect)
    ax.annotate(
        f"Window start\n$v_\\infty$={vinf_start:.2f}",
        xy=(mdates.date2num(start_launch_date), best_y),
        xytext=(20, 60),
        textcoords="offset points",
        fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec="black"),
        arrowprops=dict(arrowstyle="->"),
        annotation_clip=False,
    )
    ax.annotate(
        f"Window end\n$v_\\infty$={vinf_end:.2f}",
        xy=(mdates.date2num(end_launch_date), best_y),
        xytext=(-110, -40),
        textcoords="offset points",
        fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec="black"),
        arrowprops=dict(arrowstyle="->"),
        annotation_clip=False,
    )

    save_plot(fig, colorbar, outfile, plot_type)


def _plot_family(output_dir, epochs, stored, plot_type):
    """Generate unannotated porkchops for one cost metric."""
    for class_name, slug, title in PLOT_SPECS:
        grid = make_porkchop(stored, epochs, class_name, plot_type)
        if not _has_porkchop_data(grid):
            continue
        outfile = output_dir / Path(f"porkchop_{plot_type}_{slug}.png")
        plot_porkchop(grid, epochs, title, outfile, plot_type)

def make_plots(stored, epochs, output_dir: Path = Path("plots")):
    """Create all non-empty unannotated porkchop plot variants for the stored set."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for plot_type in ["launch", "arrival", "total", "mission"]:
        _plot_family(output_dir, epochs, stored, plot_type)


def make_annotated_plots(entries_by_class, output_dir: Path = Path("plots")):
    """Create annotated porkchops for the distinct winning trajectory classes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    spec_by_class = {class_name: (slug, title) for class_name, slug, title in PLOT_SPECS}

    for class_name, best_entry in entries_by_class.items():
        spec = spec_by_class.get(class_name)
        if spec is None:
            continue
        slug, title = spec
        window_info = {
            "best_launch": best_entry["epochs"][best_entry["traj"]["indices"][0]],
            "best_arrival": best_entry["epochs"][best_entry["traj"]["indices"][-1]],
            "window_start": best_entry["window"]["start"],
            "window_end": best_entry["window"]["stop"],
        }
        for plot_type in ["launch", "arrival", "total", "mission"]:
            grid = make_porkchop(best_entry["stored"], best_entry["epochs"], class_name, plot_type)
            if not _has_porkchop_data(grid):
                continue
            annotated_outfile = output_dir / Path(f"porkchop_{plot_type}_{slug}_annotated.png")
            plot_annotated_porkchop(
                grid, best_entry["epochs"], title, window_info, annotated_outfile, plot_type
            )


def plot_spacecraft_traj(best, epochs, bodies, tag="total", output_dir: Path = Path("plots")):
    """Save a static XY plot and animation for one selected best trajectory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (
        t0,
        tF,
        times_track,
        venus_track,
        earth_track,
        mars_track,
        jup_track,
        t_sc,
        sc_track,
    ) = prop.propagate_best_trajectory(best, epochs, bodies, n_samples=800)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(venus_track[:, 0], venus_track[:, 1], color=PLANET_COLORS["Venus"], label="Venus")
    ax.plot(earth_track[:, 0], earth_track[:, 1], color=PLANET_COLORS["Earth"], label="Earth")
    ax.plot(mars_track[:, 0], mars_track[:, 1], color=PLANET_COLORS["Mars"], label="Mars")
    ax.plot(jup_track[:, 0], jup_track[:, 1], color=PLANET_COLORS["Jupiter"], label="Jupiter")
    ax.plot(sc_track[:, 0], sc_track[:, 1], color="black", linestyle=":", label="SC path")

    body_tracks = {
        "Venus": venus_track,
        "Earth": earth_track,
        "Mars": mars_track,
        "Jupiter": jup_track,
    }
    seq_indices = [idx - best["indices"][0] for idx in best["indices"]]

    ax.scatter(earth_track[0, 0], earth_track[0, 1], color=PLANET_COLORS["Earth"], marker="o", label="Dep")
    for idx, (seq_idx, body_name) in enumerate(zip(seq_indices[1:-1], best["sequence"][1:-1])):
        body_track = body_tracks[body_name]
        ax.scatter(
            body_track[seq_idx, 0],
            body_track[seq_idx, 1],
            color=PLANET_COLORS[body_name],
            marker="o",
            s=60,
            label="GA" if idx == 0 else None,
        )
    ax.scatter(jup_track[-1, 0], jup_track[-1, 1], color=PLANET_COLORS["Jupiter"], marker="o", label="Arr")
    ax.scatter(sc_track[-1, 0], sc_track[-1, 1], color="black", marker=">", s=80, label="SC")

    ax.axis("equal")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_title(f"Best trajectory motion ({best['type']}, {t0.date()} to {tF.date()})")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / f"trajectory_plot_{tag}.png", dpi=200)
    plt.close(fig)

    animate_trajectory(
        times_track,
        t_sc,
        venus_track,
        earth_track,
        mars_track,
        jup_track,
        sc_track,
        output_dir / f"trajectory_animation_{tag}.mp4",
    )


def animate_trajectory(
    times_track,
    t_sc,
    venus_track,
    earth_track,
    mars_track,
    jup_track,
    sc_track,
    outfile,
):
    """Animate planets and spacecraft on a common heliocentric XY frame."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_title("Best trajectory animation (heliocentric XY)")

    all_x = np.hstack([venus_track[:, 0], earth_track[:, 0], mars_track[:, 0], jup_track[:, 0], sc_track[:, 0]])
    all_y = np.hstack([venus_track[:, 1], earth_track[:, 1], mars_track[:, 1], jup_track[:, 1], sc_track[:, 1]])
    pad = 0.05
    xmin, xmax = all_x.min(), all_x.max()
    ymin, ymax = all_y.min(), all_y.max()
    ax.set_xlim(xmin - pad * (xmax - xmin), xmax + pad * (xmax - xmin))
    ax.set_ylim(ymin - pad * (ymax - ymin), ymax + pad * (ymax - ymin))

    (venus_line,) = ax.plot([], [], lw=1, color=PLANET_COLORS["Venus"], label="Venus")
    (earth_line,) = ax.plot([], [], lw=1, color=PLANET_COLORS["Earth"], label="Earth")
    (mars_line,) = ax.plot([], [], lw=1, color=PLANET_COLORS["Mars"], label="Mars")
    (jup_line,) = ax.plot([], [], lw=1, color=PLANET_COLORS["Jupiter"], label="Jupiter")
    (sc_line,) = ax.plot([], [], lw=1.5, linestyle=":", color="black", label="SC path")

    (venus_pt,) = ax.plot([], [], marker="o", color=PLANET_COLORS["Venus"])
    (earth_pt,) = ax.plot([], [], marker="o", color=PLANET_COLORS["Earth"])
    (mars_pt,) = ax.plot([], [], marker="o", color=PLANET_COLORS["Mars"])
    (jup_pt,) = ax.plot([], [], marker="o", color=PLANET_COLORS["Jupiter"])
    (sc_pt,) = ax.plot([], [], marker=">", color="black")

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize=8)

    n_frames = min(300, sc_track.shape[0])
    frame_idx = np.linspace(0, sc_track.shape[0] - 1, n_frames).astype(int)
    planet_time_s = np.array(
        [(time - times_track[0]).total_seconds() for time in times_track], dtype=float
    )

    def init():
        """Reset all artists before the animation starts."""
        for obj in [
            venus_line,
            earth_line,
            mars_line,
            jup_line,
            sc_line,
            venus_pt,
            earth_pt,
            mars_pt,
            jup_pt,
            sc_pt,
        ]:
            obj.set_data([], [])
        return (
            venus_line,
            earth_line,
            mars_line,
            jup_line,
            sc_line,
            venus_pt,
            earth_pt,
            mars_pt,
            jup_pt,
            sc_pt,
        )

    def update(frame_number):
        """Advance all body and spacecraft artists to one animation frame."""
        sc_idx = frame_idx[frame_number]
        sc_time_s = t_sc[sc_idx]
        planet_idx = int(round(np.interp(sc_time_s, planet_time_s, np.arange(len(times_track)))))

        venus_line.set_data(venus_track[: planet_idx + 1, 0], venus_track[: planet_idx + 1, 1])
        earth_line.set_data(earth_track[: planet_idx + 1, 0], earth_track[: planet_idx + 1, 1])
        mars_line.set_data(mars_track[: planet_idx + 1, 0], mars_track[: planet_idx + 1, 1])
        jup_line.set_data(jup_track[: planet_idx + 1, 0], jup_track[: planet_idx + 1, 1])
        sc_line.set_data(sc_track[: sc_idx + 1, 0], sc_track[: sc_idx + 1, 1])

        venus_pt.set_data([venus_track[planet_idx, 0]], [venus_track[planet_idx, 1]])
        earth_pt.set_data([earth_track[planet_idx, 0]], [earth_track[planet_idx, 1]])
        mars_pt.set_data([mars_track[planet_idx, 0]], [mars_track[planet_idx, 1]])
        jup_pt.set_data([jup_track[planet_idx, 0]], [jup_track[planet_idx, 1]])
        sc_pt.set_data([sc_track[sc_idx, 0]], [sc_track[sc_idx, 1]])

        return (
            venus_line,
            earth_line,
            mars_line,
            jup_line,
            sc_line,
            venus_pt,
            earth_pt,
            mars_pt,
            jup_pt,
            sc_pt,
        )

    ani = FuncAnimation(fig, update, frames=n_frames, init_func=init, blit=True, interval=30)
    ani.save(outfile, writer="ffmpeg", fps=20)
    plt.close(fig)
