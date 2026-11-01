# V5 workflow (restored V3-style structure)

Keep these three files and the existing `parameters.csv` in the same folder:

- `vfarm_abm_v5.py`: model plus the baseline four-treatment factorial entry point.
- `run_sweep_v5.py`: separate 5 x 5 population-composition sweep runner.
- `parameters.csv`: shared parameter configuration (copied unchanged from the supplied V3 bundle).

## Run the factorial baseline

```powershell
python vfarm_abm_v5.py
```

Writes only:
- `factorial_timeseries_v5.csv`
- `factorial_ex_post_summary_v5.csv`

## Run the population-composition sweep

```powershell
python run_sweep_v5.py
```

Writes only:
- `population_sweep_v5.csv`
- `population_sweep_summary_v5.csv`

The default sweep is the original 5 x 5 grid for `p_C0` and `p_TypeA`, crossed with all four treatments and `S_max` seeds from `parameters.csv`. Both scripts read `T`, `S_max`, `p_rewire`, and `w` from the same parameter file by default.

For a quick smoke test (not thesis results):

```powershell
python vfarm_abm_v5.py --seeds 2 --rounds 5
python run_sweep_v5.py --seeds 2 --rounds 5
```

Do not use smoke-test outputs as final results. The scripts overwrite their respective V5 CSVs when run again. Keep V3 outputs in a separate folder as immutable references.

The new measurement fields are added to the two existing output families. No new behavioral incentive or sanction mechanism is introduced. The model explicitly reports that a separate monitoring process and graduated-sanction schedule are not implemented in V3/V5.
