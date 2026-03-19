import numpy as np
from scipy.integrate import solve_ivp

import constants as c


def twobody_sun_ode(t, y):
    """Return the heliocentric 2-body state derivative in km and km/s."""
    # y = [x,y,z,vx,vy,vz] in km and km/s
    r = y[:3]
    v = y[3:]
    rnorm = np.linalg.norm(r)
    a = -c.MU_SUN * r / (rnorm**3)
    return np.hstack((v, a))


def _sample_counts_for_legs(legs, n_samples):
    """Distribute total samples across Lambert legs while preserving endpoints."""
    if n_samples < len(legs) + 1:
        raise ValueError(
            f"n_samples={n_samples} is too small for {len(legs)} Lambert legs"
        )

    tof_days = np.array([leg["tof_days"] for leg in legs], dtype=float)
    total_tof_days = tof_days.sum()

    if total_tof_days <= 0.0:
        raise ValueError("Total time of flight must be positive")

    seg_samples = np.maximum(
        2, np.round(n_samples * tof_days / total_tof_days).astype(int)
    )

    # Each stitched segment shares one endpoint with the previous segment.
    target_total = n_samples + len(legs) - 1
    diff = int(target_total - seg_samples.sum())

    if diff != 0:
        order = np.argsort(-tof_days if diff > 0 else tof_days)
        idx = 0
        while diff != 0:
            leg_idx = order[idx % len(legs)]
            new_value = seg_samples[leg_idx] + (1 if diff > 0 else -1)
            if new_value >= 2:
                seg_samples[leg_idx] = new_value
                diff += -1 if diff > 0 else 1
            idx += 1

    return seg_samples


def _propagate_lambert_leg(leg, epochs, bodies, n_samples):
    """Propagate one Lambert leg from its departure body to its arrival body."""
    i_dep = leg["i_dep"]
    i_arr = leg["i_arr"]
    t_dep = epochs[i_dep]
    t_arr = epochs[i_arr]
    tof_s = (t_arr - t_dep).total_seconds()

    if tof_s <= 0.0:
        raise ValueError(
            f"Non-positive time of flight for leg {leg['from']}->{leg['to']}"
        )

    r0_km = bodies[leg["from"]]["pos"][i_dep] * c.AU_KM
    v0_kms = leg["v1_aud"] * c.AU_KM / c.DAY_S
    y0 = np.hstack((r0_km, v0_kms))

    t_eval = np.linspace(0.0, tof_s, n_samples)
    sol = solve_ivp(
        twobody_sun_ode, (0.0, tof_s), y0, t_eval=t_eval, rtol=1e-10, atol=1e-12
    )

    if not sol.success:
        raise RuntimeError(
            f"Propagation failed for leg {leg['from']}->{leg['to']}: {sol.message}"
        )

    return sol.t, sol.y[:3, :].T


def propagate_best_trajectory(best, epochs, bodies, n_samples=500):
    """Propagate the best direct, 1-GA, or 2-GA trajectory as stitched leg segments."""
    legs = best["lambert_legs"]
    n_legs = len(legs)

    if n_legs not in (1, 2, 3):
        raise ValueError(f"Invalid number of lambert legs: {n_legs}")

    i0 = best["indices"][0]
    iF = best["indices"][-1]
    t0 = epochs[i0]
    tF = epochs[iF]

    # Planet tracks (sample planet ephemeris at discrete indices between i0 and iF)
    idxs = np.arange(i0, iF + 1)
    venus_track = bodies["Venus"]["pos"][idxs] * c.AU_KM
    earth_track = bodies["Earth"]["pos"][idxs] * c.AU_KM
    mars_track = bodies["Mars"]["pos"][idxs] * c.AU_KM
    jup_track = bodies["Jupiter"]["pos"][idxs] * c.AU_KM
    times_track = [epochs[k] for k in idxs]

    seg_samples = _sample_counts_for_legs(legs, n_samples)

    t_segments = []
    sc_segments = []
    elapsed_offset_s = 0.0

    # Propagate each Lambert leg separately and stitch the position history together.
    for idx, (leg, leg_samples) in enumerate(zip(legs, seg_samples)):
        t_leg, sc_leg = _propagate_lambert_leg(leg, epochs, bodies, leg_samples)
        t_leg = t_leg + elapsed_offset_s
        elapsed_offset_s = t_leg[-1]

        if idx > 0:
            t_leg = t_leg[1:]
            sc_leg = sc_leg[1:]

        t_segments.append(t_leg)
        sc_segments.append(sc_leg)

    t_sc = np.concatenate(t_segments)
    sc_track = np.vstack(sc_segments)

    return (
        t0,
        tF,
        times_track,
        venus_track,
        earth_track,
        mars_track,
        jup_track,
        t_sc,
        sc_track,
    )
