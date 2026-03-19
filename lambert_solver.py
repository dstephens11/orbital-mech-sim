import multiprocessing as mp

import numpy as np
from lamberthub import gooding1990

import constants as c

# Keep and store solutions with vinf < 8 km/s at launch and/or arrival
THRESH_KMS = 8.0

_PRECOMPUTE_CTX = {}


def solve(r1, r2, t, revolutions: int, short_path: bool):
    v1, v2 = gooding1990(
        mu=c.Sun_GM,
        r1=r1,
        r2=r2,
        tof=t,
        M=revolutions,
        prograde=short_path,
        low_path=True,  # Type of solution
        maxiter=35,
        atol=1e-5,
        rtol=1e-7,
        full_output=False,  # Iteration config
    )
    return v1, v2


def lambert_solutions(r1, r2, tof, max_revs=2):
    sols = []
    if tof <= 0:
        return sols

    # Only allow multi-rev for long TOF
    rev_limit = max_revs if tof > 2 * 365 else 0
    for M in range(rev_limit + 1):
        for short in [True, False]:
            try:
                v1, v2 = solve(r1, r2, tof, M, short)
                sols.append((v1, v2))
            except:
                continue
    return sols


def flyby_is_feasible_unpowered(
    vinf_in_kms_vec,
    vinf_out_kms_vec,
    mu_km3_s2,
    r_body_km,
    hmin_km=None,
    hmax_km=None,
    vinf_match_abs_kms=None,
):
    if hmin_km is None:
        hmin_km = c.H_MIN
    if hmax_km is None:
        hmax_km = c.H_MAX
    if vinf_match_abs_kms is None:
        vinf_match_abs_kms = c.VINF_MATCH_ABS_KMS

    vinf_in = np.linalg.norm(vinf_in_kms_vec)
    vinf_out = np.linalg.norm(vinf_out_kms_vec)

    # Unpowered: magnitudes must match (within abs tolerance)
    if abs(vinf_in - vinf_out) > vinf_match_abs_kms:
        return False, None, None, None

    # Required turn angle between asymptotes
    cos_angle = np.dot(vinf_in_kms_vec, vinf_out_kms_vec) / (vinf_in * vinf_out + 1e-15)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    delta_req = np.arccos(cos_angle)  # radians

    # If essentially no turn needed, it is feasible without dipping low
    if delta_req < 1e-12:
        return True, r_body_km + hmax_km, hmax_km, delta_req

    # Required periapsis radius to achieve delta_req for hyperbola
    # rp_req = (mu / vinf^2) * (1/sin(delta/2) - 1)
    sin_half = np.sin(delta_req / 2.0)
    if sin_half <= 0.0:
        return False, None, None, delta_req

    rp_req = (mu_km3_s2 / (vinf_in**2)) * ((1.0 / sin_half) - 1.0)

    rp_min = r_body_km + hmin_km
    rp_max = r_body_km + hmax_km

    feasible = (rp_req >= rp_min) and (rp_req <= rp_max)
    if not feasible:
        return False, rp_req, rp_req - r_body_km, delta_req

    return True, rp_req, rp_req - r_body_km, delta_req


def _default_num_workers(num_departures):
    cpu_total = mp.cpu_count() or 1
    return max(1, min(num_departures, cpu_total - 1 if cpu_total > 1 else 1))


def _progress_stride(total_items):
    return max(1, total_items // 4)


def _print_progress(label, completed, total):
    pct = 100.0 * completed / total
    print(f"[{label}] {completed}/{total} departure epochs complete ({pct:.0f}%)")


def _init_precompute_worker(
    P_pos,
    P_vel,
    Q_pos,
    Q_vel,
    elapsed_days,
    from_body,
    to_body,
    max_revs,
    topk,
    max_tof_days,
):
    global _PRECOMPUTE_CTX
    _PRECOMPUTE_CTX = {
        "P_pos": P_pos,
        "P_vel": P_vel,
        "Q_pos": Q_pos,
        "Q_vel": Q_vel,
        "elapsed_days": elapsed_days,
        "from_body": from_body,
        "to_body": to_body,
        "max_revs": max_revs,
        "topk": topk,
        "max_tof_days": max_tof_days,
    }


def _precompute_departure_index(i):
    ctx = _PRECOMPUTE_CTX
    elapsed_days = ctx["elapsed_days"]
    r1 = ctx["P_pos"][i]
    vP = ctx["P_vel"][i]
    N = len(elapsed_days)

    legs = []
    for j in range(i + 1, N):
        tof_days = elapsed_days[j] - elapsed_days[i]
        if ctx["max_tof_days"] is not None and tof_days > ctx["max_tof_days"]:
            continue
        sols = lambert_solutions(r1, ctx["Q_pos"][j], tof_days, ctx["max_revs"])
        if not sols:
            continue

        for v1, v2 in sols:
            vinf_dep_aud = np.linalg.norm(v1 - vP)
            vinf_dep_kms = vinf_dep_aud * c.AU_PER_DAY_TO_KM_PER_S

            vinf_arr_aud = np.linalg.norm(v2 - ctx["Q_vel"][j])
            vinf_arr_kms = vinf_arr_aud * c.AU_PER_DAY_TO_KM_PER_S

            legs.append(
                {
                    "from": ctx["from_body"],
                    "to": ctx["to_body"],
                    "i_dep": i,
                    "i_arr": j,
                    "tof_days": float(tof_days),
                    "v1_aud": v1,
                    "v2_aud": v2,
                    "vinf_dep_kms": float(vinf_dep_kms),
                    "vinf_arr_kms": float(vinf_arr_kms),
                }
            )

    legs.sort(key=lambda d: d["vinf_dep_kms"])
    return i, legs[: ctx["topk"]]


def precompute_legs(
    bodies,
    elapsed_days,
    from_body,
    to_body,
    max_revs=2,
    topk=20,
    num_workers=None,
    max_tof_days=None,
):
    """
    Returns a dict keyed by (i_dep, i_arr) with a list of Lambert solutions.
    Also returns an outgoing-pruned dict keyed by i_dep with a list of candidate legs.
    """
    P = bodies[from_body]
    Q = bodies[to_body]

    N = len(elapsed_days)
    if N == 0:
        return {}

    if num_workers is None:
        num_workers = _default_num_workers(N)
    num_workers = max(1, min(int(num_workers), N))
    progress_label = f"{from_body}->{to_body}"
    progress_stride = _progress_stride(N)

    print(
        f"Starting Lambert precompute for {progress_label} "
        f"with {N} departure epochs on {num_workers} worker(s)..."
    )

    if num_workers == 1:
        _init_precompute_worker(
            P["pos"],
            P["vel"],
            Q["pos"],
            Q["vel"],
            elapsed_days,
            from_body,
            to_body,
            max_revs,
            topk,
            max_tof_days,
        )
        results = []
        for completed, i in enumerate(range(N), start=1):
            results.append(_precompute_departure_index(i))
            if completed == 1 or completed % progress_stride == 0 or completed == N:
                _print_progress(progress_label, completed, N)
    else:
        ctx = mp.get_context("spawn")
        chunksize = max(1, N // (num_workers * 4))
        with ctx.Pool(
            processes=num_workers,
            initializer=_init_precompute_worker,
            initargs=(
                P["pos"],
                P["vel"],
                Q["pos"],
                Q["vel"],
                elapsed_days,
                from_body,
                to_body,
                max_revs,
                topk,
                max_tof_days,
            ),
        ) as pool:
            results = []
            for completed, result in enumerate(
                pool.imap_unordered(
                    _precompute_departure_index, range(N), chunksize=chunksize
                ),
                start=1,
            ):
                results.append(result)
                if completed == 1 or completed % progress_stride == 0 or completed == N:
                    _print_progress(progress_label, completed, N)

    return dict(results)


def maybe_store(traj, stored, best, best_cost):
    cost = traj["vinf_launch_kms"] + traj["vinf_arrive_kms"]
    if cost < best_cost:
        best_cost = cost
        best = traj

    if (traj["vinf_launch_kms"] < THRESH_KMS) or (traj["vinf_arrive_kms"] < THRESH_KMS):
        stored.append(traj)

    return best, best_cost


def update_best_solutions(traj, best_by_metric):
    if (
        best_by_metric["launch"] is None
        or traj["vinf_launch_kms"] < best_by_metric["launch"]["vinf_launch_kms"]
    ):
        best_by_metric["launch"] = traj

    if (
        best_by_metric["arrival"] is None
        or traj["vinf_arrive_kms"] < best_by_metric["arrival"]["vinf_arrive_kms"]
    ):
        best_by_metric["arrival"] = traj

    traj_total_vinf = traj["vinf_launch_kms"] + traj["vinf_arrive_kms"]
    best_total = best_by_metric["total"]
    if (
        best_total is None
        or traj_total_vinf
        < best_total["vinf_launch_kms"] + best_total["vinf_arrive_kms"]
    ):
        best_by_metric["total"] = traj

    return best_by_metric


def calculate_trajectories(
    bodies,
    elapsed_days,
    max_revs=2,
    topk_direct=20,
    topk_ga=80,
    num_workers=None,
    max_years=10.0,
):
    max_tof_days = 365.25 * max_years
    print("Beginning Lambert leg generation...")
    EJ_direct = precompute_legs(
        bodies,
        elapsed_days,
        "Earth",
        "Jupiter",
        max_revs,
        topk_direct,
        num_workers,
        max_tof_days,
    )
    EJ_ga = precompute_legs(
        bodies,
        elapsed_days,
        "Earth",
        "Jupiter",
        max_revs,
        topk_ga,
        num_workers,
        max_tof_days,
    )

    EV = precompute_legs(
        bodies,
        elapsed_days,
        "Earth",
        "Venus",
        max_revs,
        topk_ga,
        num_workers,
        max_tof_days,
    )
    EM = precompute_legs(
        bodies,
        elapsed_days,
        "Earth",
        "Mars",
        max_revs,
        topk_ga,
        num_workers,
        max_tof_days,
    )

    VJ = precompute_legs(
        bodies,
        elapsed_days,
        "Venus",
        "Jupiter",
        max_revs,
        topk_ga,
        num_workers,
        max_tof_days,
    )
    MJ = precompute_legs(
        bodies,
        elapsed_days,
        "Mars",
        "Jupiter",
        max_revs,
        topk_ga,
        num_workers,
        max_tof_days,
    )

    VE = precompute_legs(
        bodies,
        elapsed_days,
        "Venus",
        "Earth",
        max_revs,
        topk_ga,
        num_workers,
        max_tof_days,
    )
    VM = precompute_legs(
        bodies,
        elapsed_days,
        "Venus",
        "Mars",
        max_revs,
        topk_ga,
        num_workers,
        max_tof_days,
    )
    MV = precompute_legs(
        bodies,
        elapsed_days,
        "Mars",
        "Venus",
        max_revs,
        topk_ga,
        num_workers,
        max_tof_days,
    )
    ME = precompute_legs(
        bodies,
        elapsed_days,
        "Mars",
        "Earth",
        max_revs,
        topk_ga,
        num_workers,
        max_tof_days,
    )
    MM = precompute_legs(
        bodies,
        elapsed_days,
        "Mars",
        "Mars",
        max_revs,
        topk_ga,
        num_workers,
        max_tof_days,
    )
    VV = precompute_legs(
        bodies,
        elapsed_days,
        "Venus",
        "Venus",
        max_revs,
        topk_ga,
        num_workers,
        max_tof_days,
    )
    print("Finished Lambert leg generation. Building trajectory combinations...")

    stored = []  # list of full trajectory dicts
    best = None
    best_cost = np.inf
    best_by_metric = {"launch": None, "arrival": None, "total": None}

    # ----------------------------
    # 0 assists (Direct)
    # ----------------------------
    for i in range(len(elapsed_days)):
        for leg in EJ_direct[i]:
            if leg["tof_days"] > max_tof_days:
                continue
            traj = {
                "type": "Direct",
                "sequence": ["Earth", "Jupiter"],
                "indices": [leg["i_dep"], leg["i_arr"]],
                "tof_days": [leg["tof_days"]],
                "lambert_legs": [leg],
                "v1_aud": leg["v1_aud"],
                "v2_aud": leg["v2_aud"],
                "vinf_launch_kms": leg["vinf_dep_kms"],
                "vinf_arrive_kms": leg["vinf_arr_kms"],
            }
            best, best_cost = maybe_store(traj, stored, best, best_cost)
            best_by_metric = update_best_solutions(traj, best_by_metric)

    # ----------------------------
    # 1 assist: Earth -> X -> Jupiter, X in {Venus, Mars}
    # ----------------------------
    first_legs = {"Venus": EV, "Mars": EM}
    second_legs = {"Venus": VJ, "Mars": MJ}

    for X in ["Venus", "Mars"]:
        L1 = first_legs[X]
        L2 = second_legs[X]
        for i in range(len(elapsed_days)):
            for leg1 in L1[i]:
                k = leg1["i_arr"]  # flyby index

                # spacecraft arrival vinf at X from leg1
                vinf_in_vec_kms = (
                    leg1["v2_aud"] - bodies[X]["vel"][k]
                ) * c.AU_PER_DAY_TO_KM_PER_S

                # next legs must depart at the same epoch k
                for leg2 in L2[k]:
                    if leg1["tof_days"] + leg2["tof_days"] > max_tof_days:
                        continue
                    j = leg2["i_arr"]
                    vinf_out_vec_kms = (
                        leg2["v1_aud"] - bodies[X]["vel"][k]
                    ) * c.AU_PER_DAY_TO_KM_PER_S

                    ok, rp_req_km, h_req_km, delta_req = flyby_is_feasible_unpowered(
                        vinf_in_vec_kms,
                        vinf_out_vec_kms,
                        bodies[X]["mu"],
                        bodies[X]["r"],
                    )
                    if not ok:
                        continue

                    traj = {
                        "type": f"{X} GA",
                        "sequence": ["Earth", X, "Jupiter"],
                        "indices": [leg1["i_dep"], k, j],
                        "tof_days": [leg1["tof_days"], leg2["tof_days"]],
                        "lambert_legs": [leg1, leg2],
                        "flyby": [
                            {
                                "body": X,
                                "index": k,
                                "vinf_in_kms": float(np.linalg.norm(vinf_in_vec_kms)),
                                "vinf_out_kms": float(np.linalg.norm(vinf_out_vec_kms)),
                                "delta_req_rad": float(delta_req),
                                "rp_req_km": (
                                    None if rp_req_km is None else float(rp_req_km)
                                ),
                                "h_req_km": (
                                    None if h_req_km is None else float(h_req_km)
                                ),
                                "h_bounds_km": [c.H_MIN, c.H_MAX],
                                "vinf_match_abs_kms": c.VINF_MATCH_ABS_KMS,
                            }
                        ],
                        "vinf_launch_kms": leg1["vinf_dep_kms"],
                        "vinf_arrive_kms": leg2["vinf_arr_kms"],
                    }
                    best, best_cost = maybe_store(traj, stored, best, best_cost)
                    best_by_metric = update_best_solutions(traj, best_by_metric)

    # ----------------------------
    # 2 assists: Earth -> X -> Y -> Jupiter
    # X in {Venus, Mars}, Y in {Earth, Venus, Mars}
    # ----------------------------
    mid_leg_map = {
        ("Venus", "Earth"): VE,
        ("Venus", "Mars"): VM,
        ("Mars", "Venus"): MV,
        ("Mars", "Earth"): ME,
        ("Mars", "Mars"): MM,
        ("Venus", "Venus"): VV,
    }
    toJ_map = {"Earth": EJ_ga, "Venus": VJ, "Mars": MJ}

    for X in ["Venus", "Mars"]:
        L1 = first_legs[X]  # Earth->X
        for Y in ["Earth", "Venus", "Mars"]:
            if (X, Y) not in mid_leg_map:
                continue
            L2 = mid_leg_map[(X, Y)]  # X->Y
            L3 = toJ_map[Y]  # Y->Jupiter

            for i in range(len(elapsed_days)):
                for leg1 in L1[i]:
                    k = leg1["i_arr"]  # encounter at X

                    vinf_in_X_vec = (
                        leg1["v2_aud"] - bodies[X]["vel"][k]
                    ) * c.AU_PER_DAY_TO_KM_PER_S

                    for leg2 in L2[k]:
                        if leg1["tof_days"] + leg2["tof_days"] > max_tof_days:
                            continue
                        m = leg2["i_arr"]  # encounter at Y

                        vinf_out_X_vec = (
                            leg2["v1_aud"] - bodies[X]["vel"][k]
                        ) * c.AU_PER_DAY_TO_KM_PER_S

                        okX, rpX, hX, dX = flyby_is_feasible_unpowered(
                            vinf_in_X_vec,
                            vinf_out_X_vec,
                            bodies[X]["mu"],
                            bodies[X]["r"],
                        )
                        if not okX:
                            continue

                        vinf_in_Y_vec = (
                            leg2["v2_aud"] - bodies[Y]["vel"][m]
                        ) * c.AU_PER_DAY_TO_KM_PER_S

                        for leg3 in L3[m]:
                            if (
                                leg1["tof_days"] + leg2["tof_days"] + leg3["tof_days"]
                                > max_tof_days
                            ):
                                continue
                            j = leg3["i_arr"]
                            vinf_out_Y_vec = (
                                leg3["v1_aud"] - bodies[Y]["vel"][m]
                            ) * c.AU_PER_DAY_TO_KM_PER_S

                            okY, rpY, hY, dY = flyby_is_feasible_unpowered(
                                vinf_in_Y_vec,
                                vinf_out_Y_vec,
                                bodies[Y]["mu"],
                                bodies[Y]["r"],
                            )
                            if not okY:
                                continue

                            traj = {
                                "type": f"{X}-{Y} GA",
                                "sequence": ["Earth", X, Y, "Jupiter"],
                                "indices": [leg1["i_dep"], k, m, j],
                                "tof_days": [
                                    leg1["tof_days"],
                                    leg2["tof_days"],
                                    leg3["tof_days"],
                                ],
                                "lambert_legs": [leg1, leg2, leg3],
                                "flyby": [
                                    {
                                        "body": X,
                                        "index": k,
                                        "vinf_in_kms": float(
                                            np.linalg.norm(vinf_in_X_vec)
                                        ),
                                        "vinf_out_kms": float(
                                            np.linalg.norm(vinf_out_X_vec)
                                        ),
                                        "delta_req_rad": float(dX),
                                        "rp_req_km": (
                                            None if rpX is None else float(rpX)
                                        ),
                                        "h_req_km": None if hX is None else float(hX),
                                        "h_bounds_km": [c.H_MIN, c.H_MAX],
                                        "vinf_match_abs_kms": c.VINF_MATCH_ABS_KMS,
                                    },
                                    {
                                        "body": Y,
                                        "index": m,
                                        "vinf_in_kms": float(
                                            np.linalg.norm(vinf_in_Y_vec)
                                        ),
                                        "vinf_out_kms": float(
                                            np.linalg.norm(vinf_out_Y_vec)
                                        ),
                                        "delta_req_rad": float(dY),
                                        "rp_req_km": (
                                            None if rpY is None else float(rpY)
                                        ),
                                        "h_req_km": None if hY is None else float(hY),
                                        "h_bounds_km": [c.H_MIN, c.H_MAX],
                                        "vinf_match_abs_kms": c.VINF_MATCH_ABS_KMS,
                                    },
                                ],
                                "vinf_launch_kms": leg1["vinf_dep_kms"],
                                "vinf_arrive_kms": leg3["vinf_arr_kms"],
                            }
                            best, best_cost = maybe_store(traj, stored, best, best_cost)
                            best_by_metric = update_best_solutions(traj, best_by_metric)

    print("Finished trajectory combination search.")
    return stored, best_by_metric, best_cost
