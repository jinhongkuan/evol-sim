# evol-sim

This repository was is linked to work that was used in the paper - *Emergence and Stability of Self-Evolved Cooperative Strategies using Stochastic Machines*. This paper was published in the IEEE Symposium Series on Computational Intelligence (SSCI) 2020.

Please note that the code is still in a maintenance state.

If you find this code or paper useful in your research, please consider citing:

Please cite this paper:

```
@inproceedings{kuan2020emergence,
  title={Emergence and Stability of Self-Evolved Cooperative Strategies using Stochastic Machines},
  author={Kuan, Jin Hong and Salecha, Aadesh},
  booktitle={2020 IEEE Symposium Series on Computational Intelligence (SSCI)},
  pages={1179--1186},
  year={2020},
  organization={IEEE}
}
```

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

## Extended Proofs

The derivations for Eq. 8 of the paper can be found [here](https://github.com/jinhongkuan/evol-sim/blob/master/Proof1.pdf).
