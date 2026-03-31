"""Jupiter arrival and orbit-insertion helpers."""

import numpy as np

import constants as c


def build_capture_orbit(
    periapsis_radius_rj=c.JOI_CAPTURE_PERIAPSIS_RJ,
    apoapsis_radius_rj=c.JOI_CAPTURE_APOAPSIS_RJ,
):
    """Build and validate an elliptical Jupiter capture-orbit definition."""
    periapsis_radius_rj = float(periapsis_radius_rj)
    apoapsis_radius_rj = float(apoapsis_radius_rj)
    if periapsis_radius_rj <= 1.0:
        raise ValueError("Capture-orbit periapsis radius must be greater than 1 Jupiter radius.")
    if apoapsis_radius_rj < periapsis_radius_rj:
        raise ValueError("Capture-orbit apoapsis radius must be greater than or equal to periapsis radius.")

    return {
        "periapsis_radius_rj": periapsis_radius_rj,
        "apoapsis_radius_rj": apoapsis_radius_rj,
        "periapsis_radius_km": periapsis_radius_rj * c.R_JUPITER,
        "apoapsis_radius_km": apoapsis_radius_rj * c.R_JUPITER,
    }


def default_capture_orbit():
    """Return the default elliptical Jupiter capture-orbit definition."""
    return build_capture_orbit()


def capture_model_summary(capture_orbit=None):
    """Return a JSON-friendly summary of an elliptical Jupiter capture model."""
    orbit = default_capture_orbit() if capture_orbit is None else capture_orbit
    return {
        "model": "elliptical_jupiter_capture",
        "periapsis_radius_rj": orbit["periapsis_radius_rj"],
        "apoapsis_radius_rj": orbit["apoapsis_radius_rj"],
        "periapsis_radius_km": orbit["periapsis_radius_km"],
        "apoapsis_radius_km": orbit["apoapsis_radius_km"],
        "periapsis_altitude_km": orbit["periapsis_radius_km"] - c.R_JUPITER,
    }


def compute_jupiter_capture(v_sc_helio_kms, v_jupiter_helio_kms, capture_orbit=None):
    """Compute Jupiter arrival ``v_inf`` and impulsive JOI into a target orbit."""
    capture_orbit = default_capture_orbit() if capture_orbit is None else capture_orbit
    v_inf_vec = np.asarray(v_sc_helio_kms, dtype=float) - np.asarray(
        v_jupiter_helio_kms, dtype=float
    )
    v_inf = float(np.linalg.norm(v_inf_vec))

    rp_km = capture_orbit["periapsis_radius_km"]
    ra_km = capture_orbit["apoapsis_radius_km"]
    a_km = 0.5 * (rp_km + ra_km)

    v_peri_hyp_kms = float(np.sqrt(v_inf**2 + 2.0 * c.MU_JUPITER / rp_km))
    v_peri_capture_kms = float(np.sqrt(c.MU_JUPITER * (2.0 / rp_km - 1.0 / a_km)))
    joi_delta_v_kms = float(v_peri_hyp_kms - v_peri_capture_kms)

    return {
        "arrival_vinf_vec_kms": v_inf_vec.tolist(),
        "arrival_vinf_kms": v_inf,
        "spacecraft_arrival_velocity_kms": np.asarray(v_sc_helio_kms, dtype=float).tolist(),
        "jupiter_arrival_velocity_kms": np.asarray(v_jupiter_helio_kms, dtype=float).tolist(),
        "hyperbolic_periapsis_speed_kms": v_peri_hyp_kms,
        "capture_periapsis_speed_kms": v_peri_capture_kms,
        "joi_delta_v_kms": joi_delta_v_kms,
        "capture_orbit": capture_orbit,
    }
