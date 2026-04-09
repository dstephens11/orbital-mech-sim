"""Earth-departure launch analysis helpers."""

import math

import constants as c


DEFAULT_PARKING_ALTITUDE_KM = 200.0
DEFAULT_LAUNCH_SITE = "Cape Canaveral"
DEFAULT_LAUNCH_LATITUDE_DEG = 28.3922


def compute_earth_departure(
    vinf_launch_kms,
    parking_altitude_km=DEFAULT_PARKING_ALTITUDE_KM,
    launch_site=DEFAULT_LAUNCH_SITE,
    launch_latitude_deg=DEFAULT_LAUNCH_LATITUDE_DEG,
):
    """Compute a simple patched-conic Earth departure from circular parking orbit.

    The reported escape-burn delta-v is the impulsive burn from circular parking orbit
    to the required hyperbolic Earth departure. The launch-site rotation benefit is
    reported separately as an ideal surface-launch assist quantity.
    """
    vinf_launch_kms = float(vinf_launch_kms)
    parking_altitude_km = float(parking_altitude_km)
    launch_latitude_deg = float(launch_latitude_deg)

    c3_km2_s2 = vinf_launch_kms**2
    parking_radius_km = c.R_EARTH + parking_altitude_km
    parking_circular_speed_kms = math.sqrt(c.MU_EARTH / parking_radius_km)
    hyperbolic_perigee_speed_kms = math.sqrt(
        vinf_launch_kms**2 + 2.0 * c.MU_EARTH / parking_radius_km
    )
    escape_burn_delta_v_kms = hyperbolic_perigee_speed_kms - parking_circular_speed_kms

    earth_rotation_benefit_kms = (
        c.OMEGA_EARTH_RAD_S * c.R_EARTH * math.cos(math.radians(launch_latitude_deg))
    )

    return {
        "model": "patched_conic_earth_departure",
        "launch_site": launch_site,
        "launch_latitude_deg": launch_latitude_deg,
        "parking_altitude_km": parking_altitude_km,
        "parking_radius_km": parking_radius_km,
        "vinf_launch_kms": vinf_launch_kms,
        "c3_km2_s2": c3_km2_s2,
        "parking_circular_speed_kms": parking_circular_speed_kms,
        "hyperbolic_perigee_speed_kms": hyperbolic_perigee_speed_kms,
        "escape_burn_delta_v_kms": escape_burn_delta_v_kms,
        "earth_rotation_benefit_kms": earth_rotation_benefit_kms,
        "notes": (
            "Escape-burn delta-v is the impulsive burn from circular parking orbit to "
            "the required hyperbolic Earth departure. The Earth-rotation term is reported "
            "separately as an ideal launch-site assist and is not subtracted from the "
            "parking-orbit escape burn."
        ),
    }
