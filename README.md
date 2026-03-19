# orbital-mech-sim

Searches heliocentric transfer opportunities from Earth to Jupiter using Lambert solutions and Sun-centered orbit propagation.

The code currently evaluates:
- direct Earth -> Jupiter transfers
- 1 gravity assist transfers using Venus or Mars
- 2 gravity assist transfers using Venus, Earth, and Mars in the allowed sequences coded in the solver

It loads planetary ephemerides from JPL Horizons, searches a launch/arrival grid, filters gravity assists using an unpowered flyby feasibility check, and then produces porkchop plots plus propagated trajectory plots and animations for the best solutions.

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

Running the code creates plots in `plots/`.

Porkchop plots:
- direct
- 1 gravity assist
- 2 gravity assists

Each is produced for:
- launch `v_inf`
- arrival `v_inf`
- total `v_inf`

Best-trajectory products:
- `plots/trajectory_plot_total.png`
- `plots/trajectory_animation_total.mp4`
- `plots/trajectory_plot_launch.png`
- `plots/trajectory_animation_launch.mp4`
- `plots/trajectory_plot_arrival.png`
- `plots/trajectory_animation_arrival.mp4`

The console output also prints summaries for the best total-, launch-, and arrival-`v_inf` trajectories.

## Configuration

The main search settings live in `run.py`:

- `MAX_REVS`: maximum Lambert revolutions
- `TOPK_DIRECT`: pruning width for direct Earth -> Jupiter legs
- `TOPK_GA`: pruning width for gravity-assist-related legs
- `NUM_WORKERS`: number of multiprocessing workers for Lambert precomputation

Search window settings:
- `start`
- `stop`
- `step`

Notes:
- `NUM_WORKERS = None` uses an automatic worker count.
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

Animation output also expects:
- `ffmpeg`

## Running

Run:

```bash
python3 run.py
```

## Current Flyby Model

Gravity assists are currently modeled as:
- unpowered flybys
- matching incoming and outgoing hyperbolic excess speed magnitude within a tolerance
- periapsis altitude constrained by `H_MIN` and `H_MAX` in `constants.py`

This is a screening model, not a full patched-conic mission design tool.

## Notes

- The Horizons query can take time for long windows.
- Lambert precomputation is the main runtime cost.
- Progress messages are printed during Lambert leg generation so long runs are easier to monitor.
