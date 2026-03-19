import numpy as np
from datetime import datetime
from astroquery.jplhorizons import Horizons

import constants as c


def parse_horizons_datetime(dt_str):
    dt_str = dt_str.strip()
    if dt_str.startswith("A.D."):
        dt_str = dt_str.replace("A.D. ", "", 1)

    return datetime.strptime(dt_str, "%Y-%b-%d %H:%M:%S.%f")


def load_bodies(start, stop, step):
    earth = Horizons(
        id="399", location="500@10", epochs={"start": start, "stop": stop, "step": step}
    ).vectors()
    venus = Horizons(
        id="299", location="500@10", epochs={"start": start, "stop": stop, "step": step}
    ).vectors()
    mars = Horizons(
        id="499", location="500@10", epochs={"start": start, "stop": stop, "step": step}
    ).vectors()
    jup = Horizons(
        id="599", location="500@10", epochs={"start": start, "stop": stop, "step": step}
    ).vectors()
    sun = Horizons(
        id="10", location="500@10", epochs={"start": start, "stop": stop, "step": step}
    ).vectors()
    print(f"Number of time steps: {len(earth)}")

    venus_pos, venus_vel = assign_state_vectors(venus)
    earth_pos, earth_vel = assign_state_vectors(earth)
    mars_pos, mars_vel = assign_state_vectors(mars)
    jup_pos, jup_vel = assign_state_vectors(jup)
    sun_pos, sun_vel = assign_state_vectors(sun)
    epochs = [parse_horizons_datetime(dt) for dt in earth["datetime_str"]]

    bodies = {
        "Sun": {"pos": sun_pos, "vel": sun_vel, "mu": None, "r": None},
        "Earth": {"pos": earth_pos, "vel": earth_vel, "mu": c.MU_EARTH, "r": c.R_EARTH},
        "Venus": {"pos": venus_pos, "vel": venus_vel, "mu": c.MU_VENUS, "r": c.R_VENUS},
        "Mars": {"pos": mars_pos, "vel": mars_vel, "mu": c.MU_MARS, "r": c.R_MARS},
        "Jupiter": {"pos": jup_pos, "vel": jup_vel, "mu": None, "r": None},
    }

    return epochs, bodies


def assign_state_vectors(horizons_obj):
    shape = (len(horizons_obj), 3)
    pos, vel = np.empty(shape), np.empty(shape)

    pos[:, 0] = np.asarray(horizons_obj["x"])
    pos[:, 1] = np.asarray(horizons_obj["y"])
    pos[:, 2] = np.asarray(horizons_obj["z"])
    vel[:, 0] = np.asarray(horizons_obj["vx"])
    vel[:, 1] = np.asarray(horizons_obj["vy"])
    vel[:, 2] = np.asarray(horizons_obj["vz"])
    return pos, vel
