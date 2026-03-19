from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

import propagate_orbit as prop


def _traj_class(traj):
    # Map your traj["type"] strings into 3 bins.
    # Adjust if your names differ.
    t = traj.get("type", "")
    if t == "Direct":
        return "Direct"
    elif (" GA" in t) and ("-" not in t):
        return "1-GA"
    elif (" GA" in t) and ("-" in t):
        return "2-GA"
    else:
        return "Other"


def make_porkchop(stored, epochs, class_name, type="launch"):
    N = len(epochs)
    Z = np.full((N, N), np.nan, dtype=float)

    for tr in stored:
        if _traj_class(tr) != class_name:
            continue
        i0 = tr["indices"][0]  # launch index
        iF = tr["indices"][-1]  # arrival index (Jupiter)
        if iF <= i0:
            continue

        if type == "launch":
            cost = tr["vinf_launch_kms"]
        elif type == "arrival":
            cost = tr["vinf_arrive_kms"]
        elif type == "total":
            cost = tr["vinf_launch_kms"] + tr["vinf_arrive_kms"]
        else:
            print(f"Invalid cost type: {type}")
            exit(1)

        # keep the best (lowest cost) value for each (launch, arrival) pair
        # f you want the same plot but colored by launch v or arrival v,
        # replace cost with tr["vinf_launch_kms"] or tr["vinf_arrive_kms"].
        if np.isnan(Z[i0, iF]) or cost < Z[i0, iF]:
            # Z[i0, iF] = cost
            Z[iF, i0] = cost

    return Z


def build_plot(Z, epochs, title):
    # Convert epochs to matplotlib date numbers
    x = mdates.date2num(epochs)  # arrival axis
    y = mdates.date2num(epochs)  # launch axis

    fig, ax = plt.subplots(figsize=(9, 7))
    X, Y = np.meshgrid(x, y)
    vmin = 5
    vmax = 25

    m = ax.pcolormesh(X, Y, Z, vmin=vmin, vmax=vmax, shading="auto")
    cb = plt.colorbar(m, ax=ax)

    ax.set_title(title)
    ax.set_xlabel("Launch date (Earth)")
    ax.set_ylabel("Arrival date (Jupiter)")

    ax.xaxis_date()
    ax.yaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    launch_start = datetime(2028, 11, 1)
    launch_end = datetime(2031, 1, 1)
    arrival_start = datetime(2031, 1, 1)
    arrival_end = datetime(2032, 1, 1)

    ax.set_xlim(mdates.date2num(launch_start), mdates.date2num(launch_end))

    ax.set_ylim(mdates.date2num(arrival_start), mdates.date2num(arrival_end))
    fig.autofmt_xdate()

    return ax, cb, arrival_start, arrival_end


def save_plot(cb, outfile, type="launch"):
    if type == "launch":
        cb.set_label(r"$v_{\infty,launch}$ (km/s)")
    elif type == "arrival":
        cb.set_label(r"$v_{\infty,arrive}$ (km/s)")
    elif type == "total":
        cb.set_label(r"$v_{\infty,launch} + v_{\infty,arrive}$ (km/s)")
    else:
        print(f"Invalid cost type: {type}")
        exit(1)

    plt.savefig(outfile, dpi=200)


def plot_porkchop(Z, epochs, title, outfile: Path, type="launch"):
    _, cb, _, _ = build_plot(Z, epochs, title)
    save_plot(cb, outfile, type)


def make_plots(stored, epochs, window_info: dict[str, datetime] = None):
    Path("plots").mkdir(parents=True, exist_ok=True)

    for plot_type in ["launch", "arrival", "total"]:
        # Build and plot three porkchops
        Z_direct = make_porkchop(stored, epochs, "Direct", plot_type)
        Z_1ga = make_porkchop(stored, epochs, "1-GA", plot_type)
        Z_2ga = make_porkchop(stored, epochs, "2-GA", plot_type)

        outfile = Path("plots") / Path(f"porkchop_{plot_type}_direct.png")
        plot_porkchop(
            Z_direct, epochs, "Porkchop: Direct Earth→Jupiter", outfile, plot_type
        )

        outfile = Path("plots") / Path(f"porkchop_{plot_type}_1ga.png")
        plot_porkchop(Z_1ga, epochs, "Porkchop: 1 Gravity Assist", outfile, plot_type)

        outfile = Path("plots") / Path(f"porkchop_{plot_type}_2ga.png")
        plot_porkchop(Z_2ga, epochs, "Porkchop: 2 Gravity Assists", outfile, plot_type)

        if window_info:
            outfile = Path("plots") / Path(f"porkchop_{plot_type}_direct_annotated.png")
            plot_annotated_direct_porkchop(
                Z_direct,
                epochs,
                "Porkchop: Direct Earth→Jupiter",
                window_info,
                outfile,
                plot_type,
            )

            outfile = Path("plots") / Path(f"porkchop_{plot_type}_1ga_annotated.png")
            plot_annotated_direct_porkchop(
                Z_1ga,
                epochs,
                "Porkchop: 1 Gravity Assist",
                window_info,
                outfile,
                plot_type,
            )

            outfile = Path("plots") / Path(f"porkchop_{plot_type}_2ga_annotated.png")
            plot_annotated_direct_porkchop(
                Z_2ga,
                epochs,
                "Porkchop: 2 Gravity Assists",
                window_info,
                outfile,
                plot_type,
            )


def plot_annotated_direct_porkchop(
    Z, epochs, title, window_info: dict[str, datetime], outfile: Path, type="launch"
):
    ax, cb, arrival_start, arrival_end = build_plot(Z, epochs, title)

    best_launch = window_info["best_launch"]
    best_arrival = window_info["best_arrival"]
    window_start = window_info["window_start"]
    window_end = window_info["window_end"]

    # convert to matplotlib date numbers
    best_x = mdates.date2num(best_launch)
    best_y = mdates.date2num(best_arrival)
    win_x0 = mdates.date2num(window_start)
    win_x1 = mdates.date2num(window_end)

    def _get_vinf(launch_date):
        try:
            # find closest launch index
            iL = np.argmin(np.abs(np.array(epochs) - launch_date))
            col = Z[:, iL]
            # return minimum ignoring NaNs
            if np.all(np.isnan(col)):
                return np.nan
            else:
                return np.nanmin(col)
        except:
            return np.nan

    vinf_best = _get_vinf(best_launch)
    vinf_start = _get_vinf(window_start)
    vinf_end = _get_vinf(window_end)
    ax.plot(
        best_x,
        best_y,
        marker="*",
        markersize=12,
        markeredgecolor="black",
        markerfacecolor="red",
        zorder=5,
    )
    ax.annotate(
        f"Best\n$v_\\infty$={vinf_best:.2f} km/s",
        xy=(best_x, best_y),
        xytext=(15, 15),
        textcoords="offset points",
        fontsize=10,
        bbox=dict(boxstyle="round", fc="white", ec="black"),
        arrowprops=dict(arrowstyle="->"),
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
        xy=(win_x0, best_y),
        xytext=(-90, 60),
        textcoords="offset points",
        fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec="black"),
        arrowprops=dict(arrowstyle="->"),
    )
    ax.annotate(
        f"Window end\n$v_\\infty$={vinf_end:.2f}",
        xy=(win_x1, best_y),
        xytext=(20, -40),
        textcoords="offset points",
        fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec="black"),
        arrowprops=dict(arrowstyle="->"),
    )

    save_plot(cb, outfile, type)


def plot_spacecraft_traj(best, epochs, bodies, tag="total"):
    """Plot and animate the stitched best trajectory using propagated spacecraft states."""
    output_dir = Path("plots")
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
    ax.plot(venus_track[:, 0], venus_track[:, 1], label="Venus (ephemeris)")
    ax.plot(earth_track[:, 0], earth_track[:, 1], label="Earth (ephemeris)")
    ax.plot(mars_track[:, 0], mars_track[:, 1], label="Mars (ephemeris)")
    ax.plot(jup_track[:, 0], jup_track[:, 1], label="Jupiter (ephemeris)")
    ax.plot(
        sc_track[:, 0], sc_track[:, 1], label="Spacecraft (stitched Sun 2-body propagation)"
    )

    body_tracks = {
        "Venus": venus_track,
        "Earth": earth_track,
        "Mars": mars_track,
        "Jupiter": jup_track,
    }
    seq_indices = [idx - best["indices"][0] for idx in best["indices"]]

    ax.scatter(sc_track[0, 0], sc_track[0, 1], marker="o", label="Launch")
    for seq_idx, body_name in zip(seq_indices[1:-1], best["sequence"][1:-1]):
        body_track = body_tracks[body_name]
        ax.scatter(
            body_track[seq_idx, 0],
            body_track[seq_idx, 1],
            marker="x",
            s=60,
            label=f"{body_name} encounter",
        )
    ax.scatter(jup_track[-1, 0], jup_track[-1, 1], marker="o", label="Arrival")

    ax.axis("equal")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_title(
        f"Best trajectory motion ({best['type']}, {t0.date()} to {tF.date()})"
    )
    ax.legend()
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
    times_track, t_sc, venus_track, earth_track, mars_track, jup_track, sc_track, outfile
):
    """Animate planets and spacecraft using propagation time to align their positions."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_title("Best trajectory animation (heliocentric XY)")

    all_x = np.hstack(
        [
            venus_track[:, 0],
            earth_track[:, 0],
            mars_track[:, 0],
            jup_track[:, 0],
            sc_track[:, 0],
        ]
    )
    all_y = np.hstack(
        [
            venus_track[:, 1],
            earth_track[:, 1],
            mars_track[:, 1],
            jup_track[:, 1],
            sc_track[:, 1],
        ]
    )
    pad = 0.05
    xmin, xmax = all_x.min(), all_x.max()
    ymin, ymax = all_y.min(), all_y.max()
    ax.set_xlim(xmin - pad * (xmax - xmin), xmax + pad * (xmax - xmin))
    ax.set_ylim(ymin - pad * (ymax - ymin), ymax + pad * (ymax - ymin))

    (venus_line,) = ax.plot([], [], lw=1, label="Venus")
    (earth_line,) = ax.plot([], [], lw=1, label="Earth")
    (mars_line,) = ax.plot([], [], lw=1, label="Mars")
    (jup_line,) = ax.plot([], [], lw=1, label="Jupiter")
    (sc_line,) = ax.plot([], [], lw=1.5, label="Spacecraft")

    (venus_pt,) = ax.plot([], [], marker="o")
    (earth_pt,) = ax.plot([], [], marker="o")
    (mars_pt,) = ax.plot([], [], marker="o")
    (jup_pt,) = ax.plot([], [], marker="o")
    (sc_pt,) = ax.plot([], [], marker="o")

    ax.legend()

    n_frames = min(300, sc_track.shape[0])
    frame_idx = np.linspace(0, sc_track.shape[0] - 1, n_frames).astype(int)
    planet_time_s = np.array(
        [(time - times_track[0]).total_seconds() for time in times_track], dtype=float
    )

    def init():
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

    def update(f):
        s = frame_idx[f]
        sc_time_s = t_sc[s]
        k = int(round(np.interp(sc_time_s, planet_time_s, np.arange(len(times_track)))))

        venus_line.set_data(venus_track[: k + 1, 0], venus_track[: k + 1, 1])
        earth_line.set_data(earth_track[: k + 1, 0], earth_track[: k + 1, 1])
        mars_line.set_data(mars_track[: k + 1, 0], mars_track[: k + 1, 1])
        jup_line.set_data(jup_track[: k + 1, 0], jup_track[: k + 1, 1])
        sc_line.set_data(sc_track[: s + 1, 0], sc_track[: s + 1, 1])

        venus_pt.set_data([venus_track[k, 0]], [venus_track[k, 1]])
        earth_pt.set_data([earth_track[k, 0]], [earth_track[k, 1]])
        mars_pt.set_data([mars_track[k, 0]], [mars_track[k, 1]])
        jup_pt.set_data([jup_track[k, 0]], [jup_track[k, 1]])
        sc_pt.set_data([sc_track[s, 0]], [sc_track[s, 1]])

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

    ani = FuncAnimation(
        fig, update, frames=n_frames, init_func=init, blit=True, interval=30
    )
    ani.save(outfile, writer="ffmpeg", fps=30)
    plt.close(fig)
