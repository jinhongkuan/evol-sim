# evol-sim
A study on the emergence of cooperation using stochastic Moore machines by [Jin Hong Kuan](https://github.com/jinhongkuan) and [Aadesh Salecha](https://github.com/AadeshSalecha) 

The scripts are written for Python 3.7.

## Executing Script
Library dependencies: `matplotlib`, `numpy`, `scipy`, `sklearn` and `xlsxwriter` 

The relevant functions are implemented in `main.py`. 

```
usage: main.py [-h] [--name NAME] [--iter ITER] [--states STATES] [--pop POP]
               [--window WINDOW] [--repetition REPETITION] [--overlap]
               [--heatmap HEATMAP]
               [--mean_score_dist MEAN_SCORE_DIST MEAN_SCORE_DIST]
               [--par_sel PAR_SEL] [--surv_sel SURV_SEL]
               [--interaction INTERACTION]
               game
```

The only required argument is the game payoff matrix. Currently, it accepts two values: `prisoner` and `no_conflict`. To define custom games, add new entries to the dictionary variable `game`.

To replicate the experiment, use 
`python main.py prisoner --default` 

The following parameters can be customized for single-run simulations:

`states`: Number of states in SMM (2 by default) 

`pop`: Number of agents in simulation (10 by default) 

`iter`: Simulation iteration count (1000 by default) 

`par_sel`: Parent/Reproductive selection method (truncation/uniform_random/roulette_wheel) 

`surv_sel`: Survival selection method (truncation/uniform_random/roulette_wheel) 

`overlap`: Toggle competition between parents and offsprings 
