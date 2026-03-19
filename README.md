# orbital-mech-sim

Searches heliocentric transfer opportunities from Earth to Jupiter using Lambert solutions and Sun-centered orbit propagation.

The code currently evaluates:
- direct Earth -> Jupiter transfers
- 1 gravity assist transfers using Venus or Mars
- 2 gravity assist transfers using Venus, Earth, and Mars

Code fetches planetary ephemerides from JPL Horizons, searches a launch/arrival grid, filters gravity assists using an unpowered flyby feasibility check, and then produces porkchop plots & propagated trajectory plots/animations for the best solutions.

## What The Code Does

Main workflow:
1. Load Sun, Venus, Earth, Mars, and Jupiter state vectors from JPL Horizons.
2. Precompute Lambert legs between relevant body pairs.
3. Search direct, 1-GA, and 2-GA trajectory combinations.
4. Track the best solutions by:
   - total v-infinity
   - launch v-infinity
   - arrival v-infinity
5. Generate porkchop plots and propagated spacecraft trajectory plots/animations.

Units:
- Lambert propagation uses AU and AU/day internally.
- Reported `v_inf` values are in km/s.
- Propagated spacecraft/planet tracks are plotted in km.

## Key Files

- `run.py`: main entry point and high-level search settings
- `horizons_reader.py`: JPL Horizons ephemeris loading
- `lambert_solver.py`: Lambert leg generation, flyby feasibility checks, and trajectory search
- `propagate_orbit.py`: stitched Sun 2-body propagation of the selected trajectory
- `plotting.py`: porkchop plots, trajectory plots, and animations
- `constants.py`: physical constants and flyby limits

## Outputs

Each run creates a new timestamped directory under `results/`, for example:

- `results/2026-03-18_15-30-12/`

That folder contains:
- `run_config.json`: the full run configuration and best-solution summary
- porkchop plots
- trajectory plots
- trajectory animations

Porkchop plots produced for launch, arrival, and total `v_inf`:
- direct
- 1 gravity assist
- 2 gravity assists

Best-trajectory products for types {launch, total, arrival}:
- `trajectory_plot_{type}.png`
- `trajectory_animation_{type}}.mp4`

The console output also prints summaries for the best total-, launch-, and arrival-`v_inf` trajectories.

## Configuration

Run settings are command-line arguments and default to the current built-in values.

Main search arguments:
- `--start`: search window start date, default `2026-07-01`
- `--stop`: search window stop date, default `2055-07-01`
- `--step`: ephemeris cadence in days, default `10`
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
- `--window-best-launch`
- `--window-best-arrival`
- `--window-start`
- `--window-end`

Notes:
- On macOS/PyCharm, the code uses multiprocessing with the `spawn` start method.
- Large search windows with fine cadence can still take significant time.

## Dependencies

The code uses:
- Python
- NumPy
- SciPy
- Matplotlib
- astroquery
- lamberthub
- ffmpeg

## Running

Run:

```bash
python3 run.py
```

Example with custom search settings:

```bash
python3 run.py --start 2028-01-01 --stop 2040-01-01 --step 20 --max-years 8 --topk-ga 120 --num-workers 8
```

Example with annotated porkchops:

```bash
python3 run.py --annotate --window-best-launch 2028-12-24 --window-best-arrival 2031-12-01 --window-start 2028-12-13 --window-end 2029-01-06
```

After the run starts, the console prints the output directory being used for that run.

## Current Flyby Model

Gravity assists are currently modeled as:
- unpowered flybys
- matching incoming and outgoing hyperbolic excess speed magnitude within a tolerance
- periapsis altitude constrained by `H_MIN` and `H_MAX` in `constants.py`

This is a screening model, not a full patched-conic mission design tool.

## Notes

- The Horizons query can take time for long windows.
- Lambert precomputation is the main runtime cost.
- Exact run settings are saved in `run_config.json`.
