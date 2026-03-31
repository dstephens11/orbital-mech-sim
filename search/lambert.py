"""Lambert leg generation and trajectory assembly routines."""

import multiprocessing as mp

import numpy as np
from lamberthub import gooding1990

from arrival import jupiter_capture as jc
import constants as c

THRESH_KMS = 8.0
_PRECOMPUTE_CTX = {}


def solve(r1, r2, t, revolutions: int, short_path: bool):
    """Solve one Lambert boundary-value problem for a specific branch."""
    v1, v2 = gooding1990(
        mu=c.Sun_GM,
        r1=r1,
        r2=r2,
        tof=t,
        M=revolutions,
        prograde=short_path,
        low_path=True,
        maxiter=35,
        atol=1e-5,
        rtol=1e-7,
        full_output=False,
    )
    return v1, v2


def lambert_solutions(r1, r2, tof, max_revs=2):
    """Return all Lambert branches considered for a departure-arrival pair."""
    sols = []
    if tof <= 0:
        return sols

    rev_limit = max_revs if tof > 2 * 365 else 0
    for revolutions in range(rev_limit + 1):
        for short_path in [True, False]:
            try:
                sols.append(solve(r1, r2, tof, revolutions, short_path))
            except Exception:
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
    """Check whether an unpowered flyby can connect two asymptotic velocity vectors."""
    if hmin_km is None:
        hmin_km = c.H_MIN
    if hmax_km is None:
        hmax_km = c.H_MAX
    if vinf_match_abs_kms is None:
        vinf_match_abs_kms = c.VINF_MATCH_ABS_KMS

    vinf_in = np.linalg.norm(vinf_in_kms_vec)
    vinf_out = np.linalg.norm(vinf_out_kms_vec)
    if abs(vinf_in - vinf_out) > vinf_match_abs_kms:
        return False, None, None, None

    cos_angle = np.dot(vinf_in_kms_vec, vinf_out_kms_vec) / (vinf_in * vinf_out + 1e-15)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    delta_req = np.arccos(cos_angle)

    if delta_req < 1e-12:
        return True, r_body_km + hmax_km, hmax_km, delta_req

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


def resolve_num_workers(num_workers=None):
    """Choose the run-level worker count for Lambert precomputation."""
    cpu_total = mp.cpu_count() or 1
    if num_workers is None:
        return max(1, cpu_total - 1 if cpu_total > 1 else 1)
    return max(1, int(num_workers))


def _progress_stride(total_items):
    """Return how often progress should be printed for a loop of this size."""
    return max(1, total_items // 2)


def _print_progress(label, completed, total):
    """Emit a simple percentage progress update for a body-pair precompute."""
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
    """Seed worker-local context so departure jobs can stay lightweight."""
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
    """Build the best Lambert legs that leave on a single departure epoch."""
    ctx = _PRECOMPUTE_CTX
    elapsed_days = ctx["elapsed_days"]
    r1 = ctx["P_pos"][i]
    vP = ctx["P_vel"][i]
    num_epochs = len(elapsed_days)

    legs = []
    for j in range(i + 1, num_epochs):
        tof_days = elapsed_days[j] - elapsed_days[i]
        if ctx["max_tof_days"] is not None and tof_days > ctx["max_tof_days"]:
            continue
        sols = lambert_solutions(r1, ctx["Q_pos"][j], tof_days, ctx["max_revs"])
        if not sols:
            continue

        for v1, v2 in sols:
            vinf_dep_aud = np.linalg.norm(v1 - vP)
            vinf_arr_aud = np.linalg.norm(v2 - ctx["Q_vel"][j])
            legs.append(
                {
                    "from": ctx["from_body"],
                    "to": ctx["to_body"],
                    "i_dep": i,
                    "i_arr": j,
                    "tof_days": float(tof_days),
                    "v1_aud": v1,
                    "v2_aud": v2,
                    "vinf_dep_kms": float(vinf_dep_aud * c.AU_PER_DAY_TO_KM_PER_S),
                    "vinf_arr_kms": float(vinf_arr_aud * c.AU_PER_DAY_TO_KM_PER_S),
                }
            )

    legs.sort(key=lambda leg: leg["vinf_dep_kms"])
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
    """Precompute pruned Lambert candidate legs for one ordered body pair."""
    P = bodies[from_body]
    Q = bodies[to_body]
    num_epochs = len(elapsed_days)
    if num_epochs == 0:
        return {}

    num_workers = resolve_num_workers(num_workers)
    progress_label = f"{from_body}->{to_body}"
    progress_stride = _progress_stride(num_epochs)

    print(
        f"Starting Lambert precompute for {progress_label} "
        f"with {num_epochs} departure epochs..."
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
        for completed, dep_index in enumerate(range(num_epochs), start=1):
            results.append(_precompute_departure_index(dep_index))
            if completed % progress_stride == 0 or completed == num_epochs:
                _print_progress(progress_label, completed, num_epochs)
    else:
        ctx = mp.get_context("spawn")
        chunksize = max(1, num_epochs // (num_workers * 4))
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
                    _precompute_departure_index, range(num_epochs), chunksize=chunksize
                ),
                start=1,
            ):
                results.append(result)
                if completed % progress_stride == 0 or completed == num_epochs:
                    _print_progress(progress_label, completed, num_epochs)

    return dict(results)


def maybe_store(traj, stored, best, best_cost):
    """Store trajectories below threshold and update the best total-cost solution."""
    cost = traj["vinf_launch_kms"] + traj["vinf_arrive_kms"]
    if cost < best_cost:
        best_cost = cost
        best = traj

    if (traj["vinf_launch_kms"] < THRESH_KMS) or (traj["vinf_arrive_kms"] < THRESH_KMS):
        stored.append(traj)

    return best, best_cost


def update_best_solutions(traj, best_by_metric):
    """Update the best launch, arrival, total, and mission trajectory trackers."""
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

    best_mission = best_by_metric["mission"]
    if (
        best_mission is None
        or traj["mission_total_cost_kms"] < best_mission["mission_total_cost_kms"]
    ):
        best_by_metric["mission"] = traj

    return best_by_metric


def _traj_signature(traj):
    """Build a stable signature used to deduplicate ranked trajectories."""
    return (
        traj["type"],
        tuple(traj["indices"]),
        round(traj["vinf_launch_kms"], 6),
        round(traj["vinf_arrive_kms"], 6),
    )


def update_ranked_trajectories(traj, ranked, limit):
    """Keep the best ``limit`` unique trajectories ranked by total ``v_inf``."""
    if limit <= 0:
        return ranked

    signature = _traj_signature(traj)
    for existing in ranked:
        if _traj_signature(existing) == signature:
            return ranked

    ranked.append(traj)
    ranked.sort(key=lambda t: t["vinf_launch_kms"] + t["vinf_arrive_kms"])
    del ranked[limit:]
    return ranked


def _record_trajectory(
    traj, stored, best, best_cost, best_by_metric, ranked, ranked_limit
):
    """Apply all bookkeeping updates for one feasible trajectory candidate."""
    best, best_cost = maybe_store(traj, stored, best, best_cost)
    best_by_metric = update_best_solutions(traj, best_by_metric)
    ranked = update_ranked_trajectories(traj, ranked, ranked_limit)
    return best, best_cost, best_by_metric, ranked


def _build_flyby_summary(
    body_name, index, vinf_in_vec, vinf_out_vec, rp_req_km, h_req_km, delta_req
):
    """Create a consistent flyby metadata block for stored trajectories."""
    return {
        "body": body_name,
        "index": index,
        "vinf_in_kms": float(np.linalg.norm(vinf_in_vec)),
        "vinf_out_kms": float(np.linalg.norm(vinf_out_vec)),
        "delta_req_rad": float(delta_req),
        "rp_req_km": None if rp_req_km is None else float(rp_req_km),
        "h_req_km": None if h_req_km is None else float(h_req_km),
        "h_bounds_km": [c.H_MIN, c.H_MAX],
        "vinf_match_abs_kms": c.VINF_MATCH_ABS_KMS,
    }


def _attach_jupiter_capture(traj, arrival_leg, arrival_index, bodies, capture_orbit=None):
    """Attach Jupiter arrival/capture metrics to a trajectory record."""
    v_sc_helio_kms = arrival_leg["v2_aud"] * c.AU_KM / c.DAY_S
    v_jup_helio_kms = bodies["Jupiter"]["vel"][arrival_index] * c.AU_KM / c.DAY_S
    arrival_analysis = jc.compute_jupiter_capture(
        v_sc_helio_kms, v_jup_helio_kms, capture_orbit=capture_orbit
    )
    traj["arrival_analysis"] = arrival_analysis
    traj["joi_delta_v_kms"] = arrival_analysis["joi_delta_v_kms"]
    traj["mission_total_cost_kms"] = (
        traj["vinf_launch_kms"] + arrival_analysis["joi_delta_v_kms"]
    )
    return traj


def _body_pair_legs(
    bodies, elapsed_days, max_revs, topk_direct, topk_ga, num_workers, max_tof_days
):
    """Precompute all ordered body-pair leg sets needed by the mission search."""
    pair_specs = {
        "EJ_direct": ("Earth", "Jupiter", topk_direct),
        "EJ_ga": ("Earth", "Jupiter", topk_ga),
        "EV": ("Earth", "Venus", topk_ga),
        "EM": ("Earth", "Mars", topk_ga),
        "VJ": ("Venus", "Jupiter", topk_ga),
        "MJ": ("Mars", "Jupiter", topk_ga),
        "VE": ("Venus", "Earth", topk_ga),
        "VM": ("Venus", "Mars", topk_ga),
        "MV": ("Mars", "Venus", topk_ga),
        "ME": ("Mars", "Earth", topk_ga),
        "MM": ("Mars", "Mars", topk_ga),
        "VV": ("Venus", "Venus", topk_ga),
    }

    legs = {}
    for name, (from_body, to_body, topk) in pair_specs.items():
        legs[name] = precompute_legs(
            bodies,
            elapsed_days,
            from_body,
            to_body,
            max_revs,
            topk,
            num_workers,
            max_tof_days,
        )
    return legs


def _build_direct_trajectories(
    bodies, EJ_direct, elapsed_days, max_tof_days, capture_orbit=None
):
    """Yield all feasible direct Earth-to-Jupiter trajectories."""
    for dep_index in range(len(elapsed_days)):
        for leg in EJ_direct[dep_index]:
            if leg["tof_days"] > max_tof_days:
                continue
            yield _attach_jupiter_capture({
                "type": "Direct",
                "sequence": ["Earth", "Jupiter"],
                "indices": [leg["i_dep"], leg["i_arr"]],
                "tof_days": [leg["tof_days"]],
                "lambert_legs": [leg],
                "v1_aud": leg["v1_aud"],
                "v2_aud": leg["v2_aud"],
                "vinf_launch_kms": leg["vinf_dep_kms"],
                "vinf_arrive_kms": leg["vinf_arr_kms"],
            }, leg, leg["i_arr"], bodies, capture_orbit=capture_orbit)


def _build_one_assist_trajectories(
    bodies, elapsed_days, EV, EM, VJ, MJ, max_tof_days, capture_orbit=None
):
    """Yield all feasible one-gravity-assist trajectories."""
    first_legs = {"Venus": EV, "Mars": EM}
    second_legs = {"Venus": VJ, "Mars": MJ}

    for flyby_body in ["Venus", "Mars"]:
        first_set = first_legs[flyby_body]
        second_set = second_legs[flyby_body]
        for dep_index in range(len(elapsed_days)):
            for leg1 in first_set[dep_index]:
                flyby_index = leg1["i_arr"]
                vinf_in_vec = (
                    leg1["v2_aud"] - bodies[flyby_body]["vel"][flyby_index]
                ) * c.AU_PER_DAY_TO_KM_PER_S

                for leg2 in second_set[flyby_index]:
                    if leg1["tof_days"] + leg2["tof_days"] > max_tof_days:
                        continue
                    arr_index = leg2["i_arr"]
                    vinf_out_vec = (
                        leg2["v1_aud"] - bodies[flyby_body]["vel"][flyby_index]
                    ) * c.AU_PER_DAY_TO_KM_PER_S

                    ok, rp_req_km, h_req_km, delta_req = flyby_is_feasible_unpowered(
                        vinf_in_vec,
                        vinf_out_vec,
                        bodies[flyby_body]["mu"],
                        bodies[flyby_body]["r"],
                    )
                    if not ok:
                        continue

                    yield _attach_jupiter_capture({
                        "type": f"{flyby_body} GA",
                        "sequence": ["Earth", flyby_body, "Jupiter"],
                        "indices": [leg1["i_dep"], flyby_index, arr_index],
                        "tof_days": [leg1["tof_days"], leg2["tof_days"]],
                        "lambert_legs": [leg1, leg2],
                        "flyby": [
                            _build_flyby_summary(
                                flyby_body,
                                flyby_index,
                                vinf_in_vec,
                                vinf_out_vec,
                                rp_req_km,
                                h_req_km,
                                delta_req,
                            )
                        ],
                        "vinf_launch_kms": leg1["vinf_dep_kms"],
                        "vinf_arrive_kms": leg2["vinf_arr_kms"],
                    }, leg2, arr_index, bodies, capture_orbit=capture_orbit)


def _build_two_assist_trajectories(
    bodies,
    elapsed_days,
    EV,
    EM,
    EJ_ga,
    VJ,
    MJ,
    VE,
    VM,
    MV,
    ME,
    MM,
    VV,
    max_tof_days,
    capture_orbit=None,
):
    """Yield all feasible two-gravity-assist trajectories."""
    first_legs = {"Venus": EV, "Mars": EM}
    mid_leg_map = {
        ("Venus", "Earth"): VE,
        ("Venus", "Mars"): VM,
        ("Mars", "Venus"): MV,
        ("Mars", "Earth"): ME,
        ("Mars", "Mars"): MM,
        ("Venus", "Venus"): VV,
    }
    toJ_map = {"Earth": EJ_ga, "Venus": VJ, "Mars": MJ}

    for first_body in ["Venus", "Mars"]:
        first_set = first_legs[first_body]
        for second_body in ["Earth", "Venus", "Mars"]:
            if (first_body, second_body) not in mid_leg_map:
                continue

            second_set = mid_leg_map[(first_body, second_body)]
            third_set = toJ_map[second_body]

            for dep_index in range(len(elapsed_days)):
                for leg1 in first_set[dep_index]:
                    first_index = leg1["i_arr"]
                    vinf_in_first = (
                        leg1["v2_aud"] - bodies[first_body]["vel"][first_index]
                    ) * c.AU_PER_DAY_TO_KM_PER_S

                    for leg2 in second_set[first_index]:
                        if leg1["tof_days"] + leg2["tof_days"] > max_tof_days:
                            continue
                        second_index = leg2["i_arr"]
                        vinf_out_first = (
                            leg2["v1_aud"] - bodies[first_body]["vel"][first_index]
                        ) * c.AU_PER_DAY_TO_KM_PER_S

                        ok_first, rp_first, h_first, delta_first = (
                            flyby_is_feasible_unpowered(
                                vinf_in_first,
                                vinf_out_first,
                                bodies[first_body]["mu"],
                                bodies[first_body]["r"],
                            )
                        )
                        if not ok_first:
                            continue

                        vinf_in_second = (
                            leg2["v2_aud"] - bodies[second_body]["vel"][second_index]
                        ) * c.AU_PER_DAY_TO_KM_PER_S

                        for leg3 in third_set[second_index]:
                            total_tof = (
                                leg1["tof_days"] + leg2["tof_days"] + leg3["tof_days"]
                            )
                            if total_tof > max_tof_days:
                                continue
                            arr_index = leg3["i_arr"]
                            vinf_out_second = (
                                leg3["v1_aud"]
                                - bodies[second_body]["vel"][second_index]
                            ) * c.AU_PER_DAY_TO_KM_PER_S

                            ok_second, rp_second, h_second, delta_second = (
                                flyby_is_feasible_unpowered(
                                    vinf_in_second,
                                    vinf_out_second,
                                    bodies[second_body]["mu"],
                                    bodies[second_body]["r"],
                                )
                            )
                            if not ok_second:
                                continue

                            yield _attach_jupiter_capture({
                                "type": f"{first_body}-{second_body} GA",
                                "sequence": [
                                    "Earth",
                                    first_body,
                                    second_body,
                                    "Jupiter",
                                ],
                                "indices": [
                                    leg1["i_dep"],
                                    first_index,
                                    second_index,
                                    arr_index,
                                ],
                                "tof_days": [
                                    leg1["tof_days"],
                                    leg2["tof_days"],
                                    leg3["tof_days"],
                                ],
                                "lambert_legs": [leg1, leg2, leg3],
                                "flyby": [
                                    _build_flyby_summary(
                                        first_body,
                                        first_index,
                                        vinf_in_first,
                                        vinf_out_first,
                                        rp_first,
                                        h_first,
                                        delta_first,
                                    ),
                                    _build_flyby_summary(
                                        second_body,
                                        second_index,
                                        vinf_in_second,
                                        vinf_out_second,
                                        rp_second,
                                        h_second,
                                        delta_second,
                                    ),
                                ],
                                "vinf_launch_kms": leg1["vinf_dep_kms"],
                                "vinf_arrive_kms": leg3["vinf_arr_kms"],
                            }, leg3, arr_index, bodies, capture_orbit=capture_orbit)


def calculate_trajectories(
    bodies,
    elapsed_days,
    max_revs=2,
    topk_direct=20,
    topk_ga=80,
    num_workers=None,
    max_years=10.0,
    ranked_limit=50,
    capture_orbit=None,
):
    """Search direct, one-assist, and two-assist Earth-to-Jupiter trajectories."""
    max_tof_days = 365.25 * max_years
    print("Beginning Lambert leg generation...")
    legs = _body_pair_legs(
        bodies,
        elapsed_days,
        max_revs,
        topk_direct,
        topk_ga,
        num_workers,
        max_tof_days,
    )
    print("Finished Lambert leg generation. Building trajectory combinations...")

    stored = []
    best = None
    best_cost = np.inf
    best_by_metric = {"launch": None, "arrival": None, "total": None, "mission": None}
    ranked = []

    trajectory_sources = [
        _build_direct_trajectories(
            bodies, legs["EJ_direct"], elapsed_days, max_tof_days, capture_orbit
        ),
        _build_one_assist_trajectories(
            bodies,
            elapsed_days,
            legs["EV"],
            legs["EM"],
            legs["VJ"],
            legs["MJ"],
            max_tof_days,
            capture_orbit,
        ),
        _build_two_assist_trajectories(
            bodies,
            elapsed_days,
            legs["EV"],
            legs["EM"],
            legs["EJ_ga"],
            legs["VJ"],
            legs["MJ"],
            legs["VE"],
            legs["VM"],
            legs["MV"],
            legs["ME"],
            legs["MM"],
            legs["VV"],
            max_tof_days,
            capture_orbit,
        ),
    ]

    for source in trajectory_sources:
        for traj in source:
            best, best_cost, best_by_metric, ranked = _record_trajectory(
                traj,
                stored,
                best,
                best_cost,
                best_by_metric,
                ranked,
                ranked_limit,
            )

    print("Finished trajectory combination search.")
    return stored, best_by_metric, best_cost, ranked
