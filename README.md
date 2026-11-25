# Engine health at taxi power: feasibility probe

A speed-governed engine hides wear at low power: the control system holds speed by burning more fuel, so wear shows up as fuel flow and exhaust temperature. Bleed air and accessory load have the same effect, and weigh more at taxi power.

This is an exploratory probe, not a finished benchmark or an engine model validated against tests. All data are synthetic. No C-MAPSS and no operator or manufacturer data are used.

- probe.py: static single-spool turbojet with compressor map, choked or unchoked turbine and nozzle, and three known wear parameters; scores simple baselines.
- map_check.py: physics fit with the exact simulator, priors and convergence check. It supersedes the "MAP" rows of probe.py, which use wrong prior means and stop at 25 evaluations.
- results_*.txt: stored output of both scripts.

The engine parameters and noise levels are illustrative. The metric, 1 - RMSE/RMSE(constant) clipped to [0, 1], is provisional.

Run: `pip install -r requirements.txt`, then `python probe.py` and `python map_check.py` (Python 3.12).
