# GTA VI Twitch CCU Forecaster

Entry for make.fun competition `gta6-twitch-ccu-forecast-01`: forecast Twitch
concurrent viewers in the Grand Theft Auto VI category for hours 2–24 of
release day (19 Nov 2026 UTC), five quantiles per hour, scored by pinball
loss on the log scale.

## Model

`main.ts` (Node 22+ stdlib only, no network, deterministic) fits a shape
model from the case file's own launch comparables at runtime:

- For each comparable, take its 24 release-day hourly prints. Drop
  comparables with no release day in the window, a missing print, or hour-1
  viewers below 100.
- For each forecast hour, kernel-weighted least squares of
  `log(viewers_h)` on `log(viewers_1)`, with a Gaussian kernel (0.6 dex
  bandwidth) on distance in log hour-1 size — so the fit is driven by past
  launches whose hour-1 print was close to this case's. This matters because
  the day's shape depends on hour-1 size: trickle hour-1s grow 100–1000x,
  huge hour-1s only ~1–3x.
- Quantiles from the weighted residual SD (×0.8, practice-calibrated) around
  the fitted log-mean, with standard-normal z at 0.05/0.25/0.5/0.75/0.95.
- Defensive fallbacks (persistence shape) keep every case valid instead of
  crashing.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

## Practice results (4 past launches, answers from SullyGnome hourly averages)

| case | starter (persistence) | this model |
| --- | --- | --- |
| Marvel's Wolverine | 0.744 | 0.181 |
| Red Dead Redemption 2 | 0.287 | 0.214 |
| God of War Ragnarök | 0.215 | 0.138 |
| Marvel's Spider-Man 2 | 0.303 | 0.184 |
| **total** | **0.387** | **0.179** |

Typical miss 33% (starter 83%), 90% coverage 0.935. Runtime ~0.16 s/case,
two runs byte-identical.
