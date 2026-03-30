# orbital-mech-sim

Searches heliocentric transfer opportunities from Earth to Jupiter using Lambert solutions and Sun-centered orbit propagation.

The code currently evaluates:
- direct Earth -> Jupiter transfers
- 1 gravity assist transfers using Venus or Mars
- 2 gravity assist transfers using Venus, Earth, and Mars

Code fetches planetary ephemerides from JPL Horizons, searches a launch/arrival grid, filters gravity assists using an unpowered flyby feasibility check, and then produces porkchop plots & propagated trajectory plots/animations for the best solutions.

The current implementation also models an impulsive Jupiter orbit insertion into a fixed elliptical capture orbit:
- periapsis = `1.1 Rj`
- apoapsis = `115 Rj`

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
- `ephemeris.py`: JPL Horizons ephemeris loading
- `search/lambert.py`: Lambert leg generation and trajectory assembly
- `search/refinement.py`: adaptive coarse-to-fine search logic
- `dynamics/propagation.py`: stitched Sun 2-body propagation
- `visualization/plots.py`: porkchop plots, trajectory plots, and animations
- `constants.py`: physical constants and flyby limits

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

`run.py` is the single user-facing entry point. It handles argument parsing, multiprocessing setup, the adaptive search workflow, and output generation.

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

## Current Jupiter Capture Model

Jupiter arrival is currently modeled as:
- a hyperbolic arrival defined by the final Lambert-leg Jupiter-relative `v_inf`
- an impulsive JOI burn at periapsis
- capture into a fixed elliptical orbit with periapsis `1.1 Rj` and apoapsis `115 Rj` (based on JUNO spacecraft orbit)

The code reports both:
- arrival `v_inf`
- JOI delta-v

and uses `launch v_inf + JOI delta-v` as a first-class mission metric alongside the legacy `v_inf` metrics.

## Notes

- The Horizons query can take time for long windows.
- Lambert precomputation is the main runtime cost.
- Exact run settings are saved in `run_config.json`.
- The trajectory/ephemeris snapshot files can be reused for downstream JOI and mission-analysis work without rerunning the full search.
