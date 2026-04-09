# orbital-mech-sim

Searches heliocentric transfer opportunities from Earth to Jupiter using Lambert solutions and Sun-centered orbit propagation.

The code currently evaluates:
- direct Earth -> Jupiter transfers
- 1 gravity assist transfers using Venus or Mars
- 2 gravity assist transfers using Venus, Earth, and Mars

Code fetches planetary ephemerides from JPL Horizons, searches a launch/arrival grid, filters gravity assists using an unpowered flyby feasibility check, and then produces porkchop plots & propagated trajectory plots/animations for the best solutions.

The current implementation also models an impulsive Jupiter orbit insertion into an elliptical capture orbit that defaults to:
- periapsis = `1.1 Rj`
- apoapsis = `115 Rj`

A separate downstream script can now re-run the Jupiter capture analysis from a saved snapshot using a configurable target capture orbit, without rerunning the Lambert search.
That downstream analysis now also produces a local Jupiter-centered capture plot and animation, with the Sun direction shown when the sibling NPZ snapshot is available.

The search now supports adaptive refinement:
- a coarse global search over the full date span
- automatic follow-on searches in narrower windows around the best candidate corridors
- optional annotated porkchops driven by the final refined window automatically

## What The Code Does

Main workflow:
1. Load Sun, Venus, Earth, Mars, and Jupiter state vectors from JPL Horizons.
2. Precompute Lambert legs between relevant body pairs.
3. Search direct, 1-GA, and 2-GA trajectory combinations.
4. Refine the best candidate corridors on denser date grids.
5. Track the best solutions by:
   - total v-infinity
   - launch + JOI
   - launch v-infinity
   - arrival v-infinity
6. Generate porkchop plots and propagated spacecraft trajectory plots/animations.

Units:
- Lambert propagation uses AU and AU/day internally.
- Reported `v_inf` values are in km/s.
- Reported JOI costs are in km/s.
- Propagated spacecraft/planet tracks are plotted in km.

## Key Files

- `run.py`: main entry point
- `jupiter_capture_from_snapshot.py`: standalone JOI/capture reanalysis from a saved snapshot
- `delta_v_budget.py`: standalone mission delta-v budgeting from a saved results folder or snapshot
- `ephemeris.py`: JPL Horizons ephemeris loading
- `search/lambert.py`: Lambert leg generation and trajectory assembly
- `search/refinement.py`: adaptive coarse-to-fine search logic
- `dynamics/propagation.py`: stitched Sun 2-body propagation
- `visualization/plots.py`: porkchop plots, trajectory plots, and animations
- `visualization/capture.py`: Jupiter-centered JOI/capture plots and animations
- `constants.py`: physical constants and flyby limits
- `snapshot_io.py`: shared snapshot serialization/loading helpers

## Outputs

Each run creates a new timestamped directory under `results/`, for example:

- `results/2026-03-18_15-30-12/`

That folder contains:
- `run_config.json`: the full run configuration and best-solution summary
- `mission_design_report.md`: human-readable searched-window, optimal-trajectory, and JOI/capture summary
- `trajectory_ephemeris_snapshot.json`: reusable best-trajectory snapshot with Lambert leg data
- `trajectory_ephemeris_snapshot.npz`: reusable ephemeris arrays for the best trajectories
- adaptive search level summaries and the final annotated window
- porkchop plots
- trajectory plots
- trajectory animations
- optional downstream capture reanalysis products such as `jupiter_capture_mission.json`
- optional downstream mission budget products such as `delta_v_budget_mission.json`
- optional Jupiter-centered capture products such as `jupiter_capture_plot_mission.png`
- optional Jupiter-centered capture animations such as `jupiter_capture_animation_mission.mp4`

`run_config.json` and `mission_design_report.md` now also include the fixed Jupiter capture-orbit definition plus JOI/capture details such as arrival `v_inf`, JOI delta-v, and capture-orbit periapsis information.

Porkchop plots produced for launch, arrival, total `v_inf`, and launch + JOI:
- direct
- 1 gravity assist
- 2 gravity assists

Best-trajectory products for types {launch, total, arrival, mission}:
- `trajectory_plot_{type}.png`
- `trajectory_animation_{type}}.mp4`

The console output also prints summaries for the best total-, mission-, launch-, and arrival-metric trajectories.
Here `mission` means `launch v_inf + JOI delta-v`.

## Configuration

Run settings are command-line arguments and default to the current built-in values.

Main search arguments:
- `--start`: search window start date, default `2026-07-01`
- `--stop`: search window stop date, default `2055-07-01`
- `--step`: coarse global ephemeris cadence in days, default `10`
- `--refine-steps`: comma-separated refinement cadences; default halves the coarse step down to `1`
- `--refine-top-n`: number of best corridors advanced to the next level, default `3`
- `--refine-pad-scale`: half-width of each refined window in units of the next cadence, default `12.0`
- `--max-years`: maximum total trajectory duration in years, default `10`
- `--max-revs`: maximum Lambert revolutions, default `2`
- `--topk-direct`: pruning width for direct Earth -> Jupiter legs, default `20`
- `--topk-ga`: pruning width for gravity-assist-related legs, default `80`
- `--num-workers`: multiprocessing worker count, default auto-select
- `--thresh-kms`: storage threshold for launch or arrival `v_inf` in km/s, default `8.0`

Flyby-screening arguments:
- `--h-min-km`: minimum flyby altitude in km
- `--h-max-km`: maximum flyby altitude in km
- `--vinf-match-abs-kms`: allowed incoming/outgoing `v_inf` magnitude mismatch in km/s
- `--joi-periapsis-rj`: target Jupiter capture-orbit periapsis radius in Jupiter radii, default `1.1`
- `--joi-apoapsis-rj`: target Jupiter capture-orbit apoapsis radius in Jupiter radii, default `115`

Annotated porkchop arguments:
- `--annotate`: enable annotated porkchop plots

When `--annotate` is enabled, the annotation rectangle is taken from the final refined search window automatically. The best-launch and best-arrival markers come from the best total-`v_inf` trajectory found by the adaptive search.

Notes:
- On macOS/PyCharm, the code uses multiprocessing with the `spawn` start method.
- Large search windows with fine cadence can still take significant time.
- The adaptive refinement flow is intended to replace the old manual coarse-run then fine-run workflow.

## Dependencies

The code uses:
- Python
- NumPy
- SciPy
- Matplotlib
- astroquery
- lamberthub
- ffmpeg

## Setup

The recommended setup is a conda environment because `lamberthub` pulls in `numba` and `llvmlite`, which have been more reliable to install this way on macOS.

Create and activate the environment:

```bash
conda create -n orbital python=3.11 numpy scipy matplotlib astroquery numba llvmlite
conda activate orbital
pip install lamberthub
```

If you prefer, you can still use `requirements.txt` as a package reference, but the conda-based install above is the recommended path for this project.

## Running

Run:

```bash
python3 run.py
```

`run.py` is the main search entry point. It handles argument parsing, multiprocessing setup, the adaptive search workflow, and output generation.

By default, both `run.py` and `jupiter_capture_from_snapshot.py` use a Jupiter capture orbit of `1.1 Rj x 115 Rj`. Both entry points let you override that orbit if you already know the target capture geometry you want to analyze.

For downstream Jupiter-capture-only work from a saved snapshot:

```bash
python3 jupiter_capture_from_snapshot.py results/<run>/trajectory_ephemeris_snapshot.json
```

Example with a different target capture orbit and a different saved best trajectory:

```bash
python3 jupiter_capture_from_snapshot.py \
  results/<run>/trajectory_ephemeris_snapshot.json \
  --best arrival \
  --periapsis-rj 1.2 \
  --apoapsis-rj 80
```

That script writes:
- `jupiter_capture_<best>.json`
- `jupiter_capture_<best>.md`
- `jupiter_capture_plot_<best>.png`
- `jupiter_capture_animation_<best>.mp4`

The Jupiter-centered plot and animation are representative planar capture views derived from the saved arrival `v_inf` and target capture orbit. When the sibling NPZ snapshot is present, the local plot also shows the Jupiter-to-Sun direction from the saved arrival ephemeris and orients the approximate capture ellipse with periapsis Sunward and apoapsis anti-Sunward by convention.

For downstream mission delta-v budgeting from the latest results folder:

```bash
python3 delta_v_budget.py
```

Example with an explicit results folder and custom budget assumptions:

```bash
python3 delta_v_budget.py results/<run> \
  --best mission \
  --margin-pct 15 \
  --post-launch-tcm-ms 40 \
  --pre-flyby-tcm-ms 20 \
  --jupiter-cruise-tcm-ms-per-year 15 \
  --jupiter-ops-years 2.0 \
  --stationkeeping-ms-per-year 20
```

That script writes:
- `delta_v_budget_<best>.json`
- `delta_v_budget_<best>.md`

The budget uses the saved launch `v_inf` and JOI values from the selected trajectory, then layers configurable planning allowances for post-launch corrections, flyby cleanup, Jupiter-leg cruise corrections, post-JOI cleanup, Jupiter-orbit operations, and an end-of-mission disposal burn, all with one flat configurable margin percentage.
All outputs from `delta_v_budget.py` are reported in `m/s`.
The launch row now uses a simple patched-conic Earth departure from a circular parking orbit, and the selected-trajectory summary also reports launch `C3` and required escape-burn delta-v.

Example with custom search settings:

```bash
python3 run.py --start 2028-01-01 --stop 2040-01-01 --step 20 --refine-steps 10,5,1 --max-years 8 --topk-ga 120 --num-workers 8
```

Example with annotated porkchops:

```bash
python3 run.py --annotate --step 20 --refine-steps 10,5,1
```

After the run starts, the console prints the output directory being used for that run.

## Current Flyby Model

Gravity assists are currently modeled as:
- unpowered flybys
- matching incoming and outgoing hyperbolic excess speed magnitude within a tolerance
- periapsis altitude constrained by `H_MIN` and `H_MAX` in `constants.py`

This is a screening model, not a full patched-conic mission design tool.

## Current Launch Model

Earth departure is currently modeled as:
- a simple patched-conic departure from a circular `200 km` parking orbit
- launch from Cape Canaveral latitude
- launch `C3 = v_inf^2`
- required hyperbolic escape-burn delta-v from parking orbit
- Earth-rotation benefit reported separately as an ideal launch-site assist term

This is more realistic than using heliocentric departure `v_inf` alone, but it is still not a full ascent or launcher-performance model.

## Current Jupiter Capture Model

Jupiter arrival is currently modeled as:
- a hyperbolic arrival defined by the final Lambert-leg Jupiter-relative `v_inf`
- an impulsive JOI burn at periapsis
- capture into an elliptical orbit with default periapsis `1.1 Rj` and default apoapsis `115 Rj` (based on JUNO spacecraft orbit)

The code reports both:
- arrival `v_inf`
- JOI delta-v

and uses `launch v_inf + JOI delta-v` as a first-class mission metric alongside the legacy `v_inf` metrics.

The search pipeline uses that default capture orbit for mission ranking unless you override it with `--joi-periapsis-rj` and `--joi-apoapsis-rj`. You can also re-evaluate the same saved arrival state against a different target Jupiter orbit with `jupiter_capture_from_snapshot.py`.

The downstream Jupiter-centered capture visualization is intentionally approximate:
- it is derived from the saved arrival `v_inf` vector and target capture orbit, not a full patched-conic arrival solve
- the Sun marker is taken from Jupiter's saved heliocentric arrival position when `trajectory_ephemeris_snapshot.npz` is available
- the displayed capture ellipse is oriented by convention so periapsis is Sunward and apoapsis is anti-Sunward

## Notes

- The Horizons query can take time for long windows.
- Lambert precomputation is the main runtime cost.
- Exact run settings are saved in `run_config.json`.
- The trajectory/ephemeris snapshot files can be reused for downstream JOI and mission-analysis work without rerunning the full search.
