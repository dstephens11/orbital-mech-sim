"""
----------------------------
Define physical constants
----------------------------
Keep Lambert propagation in AU/day (Sun_GM in AU^3/day^2),
but do flyby and thresholding in km/s for easier intuition.
"""

Sun_GM = 2.959122e-4  # AU^3/day^2
AU_KM = 149597870.700  # km
DAY_S = 86400.0  # s
AU_PER_DAY_TO_KM_PER_S = AU_KM / DAY_S  # ~1731.456...

# Flyby physics constants (km units)
MU_EARTH = 398600.4418
MU_VENUS = 324858.592
MU_MARS = 42828.375214
MU_SUN = Sun_GM * (AU_KM**3) / (DAY_S**2)

R_EARTH = 6378.1363
R_VENUS = 6051.8
R_MARS = 3396.19

# Flyby altitude sweep (km above body radius)
H_MIN = 200.0
H_MAX = 300000.0

# Unpowered flyby requirement: | |vinf_in| - |vinf_out| | <= 0.2 km/s
VINF_MATCH_ABS_KMS = 0.2
