# Literature → router weights (candidate)

Labels used (medium/high confidence, non-none): **216**

## Label agreement (approximate contexts)

- baseline (default rules): 0.352
- candidate weights: 0.497
- held-out split: 0.460

### By space type

- mixed: 0.447
- categorical_heavy: 0.214
- continuous_heavy: 0.573
- unknown: 0.875

## Benchmark LOPO check (runs/calibration_180/records.json)

- baseline top1/top2: 0.167 / 0.833
- candidate top1/top2: 0.75 / 0.833

## Candidate weights

```json
{
  "bayes": 1.535,
  "constrained": 1.536,
  "grid": 0.631,
  "lhs": 1.039,
  "mfbo": 0.736,
  "parego": 0.956,
  "qehvi": 0.713,
  "qnehvi": 0.722,
  "random": 1.443,
  "sobol": 0.618,
  "tpe": 1.146
}
```

## How to use

```powershell
py -3 scripts/run_condopt.py route --space runs/prepared/space.json --weights configs/router_weights_literature.json --json
```

Not enabled by default: label contexts are approximate (abstract-only), so
this is a prior to test, not a replacement for benchmark/default rules.