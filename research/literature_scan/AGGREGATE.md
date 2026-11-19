# Literature scan — router experience

Generated: 2026-11-05T20:48:10+08:00
Papers: 1357

## Coverage

- algorithm_terms: 32.3%
- automation_tags: 16.4%
- abstract: 88.4%

## Algorithm mentions

- Bayesian optimization: 217
- Neural network: 89
- TPE: 68
- Multi-objective: 66
- Gradient boosting: 36
- Random forest: 35
- Gaussian process: 35
- Active learning: 34
- RSM: 34
- Reinforcement learning: 27
- Pareto: 24
- HTE: 23
- Genetic algorithm: 17
- Orthogonal design: 14
- DoE: 13
- PSO: 9
- LHS: 5
- Factorial design: 3
- Sobol: 3
- qNEHVI: 1
- EI / Bayesian optimization: 1

## Automation mentions

- closed loop: 93
- robotic: 68
- active learning: 34
- self-driving lab: 32
- HTE: 23
- flow: 17
- autonomous experimentation: 11
- autonomous lab: 1
- self-optimizing: 1

## Subsets
### categorical_heavy (n=4)

- TPE: 1
- Bayesian optimization: 1

### continuous_heavy (n=1)

- Bayesian optimization: 1
- Multi-objective: 1
- Pareto: 1

### autonomous (n=222)

- Bayesian optimization: 38
- Active learning: 34
- Multi-objective: 26
- HTE: 23
- Neural network: 16
- Reinforcement learning: 12
- Gaussian process: 9
- Pareto: 9
- TPE: 8
- Random forest: 6
- DoE: 4
- Orthogonal design: 3
- Factorial design: 2
- Gradient boosting: 2
- RSM: 2
- qNEHVI: 1
- Sobol: 1
- PSO: 1
- Genetic algorithm: 1

### multi_objective (n=104)

- Multi-objective: 66
- Bayesian optimization: 35
- Pareto: 24
- Active learning: 12
- TPE: 8
- RSM: 6
- Random forest: 5
- Gaussian process: 5
- Gradient boosting: 5
- Genetic algorithm: 5
- Neural network: 4
- HTE: 3
- Reinforcement learning: 3
- qNEHVI: 1
- Sobol: 1
- PSO: 1
- DoE: 1
- LHS: 1
- Orthogonal design: 1
- EI / Bayesian optimization: 1

### constraints (n=315)

- Bayesian optimization: 56
- Multi-objective: 29
- Neural network: 22
- TPE: 22
- Reinforcement learning: 14
- Active learning: 13
- Pareto: 9
- Gaussian process: 8
- Gradient boosting: 8
- Random forest: 6
- Genetic algorithm: 6
- RSM: 5
- HTE: 4
- Sobol: 3
- DoE: 2
- PSO: 2

## Reproduce

```powershell
py -3 scripts/literature_scan.py all --config configs/literature_scan.json
```

Caveat: rule-based extraction from titles/abstracts only; counts are
mentions, not experimental superiority, and abstracts often omit details.