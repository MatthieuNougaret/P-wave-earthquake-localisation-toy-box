# P-wave-earthquake-localisation-toy-box
Toy box to test various earthquakes localisation algorithms using P-wave arrival time into a simplified world.

## The world

The world is a simplified toy box which is both homogeneous and isotropic.

By default, there are 5 seismic stations that are almost coplanar and are distributed slightly asymmetrically in the form of a diamond or a kite.


## Solvers

#### Gauss-Newton:
This solver use Gauss-Newton method to search for the earthquake hypocenter and failure timing.

It incorporate a semi-automatic dampling method to control step size and a reset control to keep the search in the ground (Z <= 0).

An early stoping is raised when the steps are smaller than $1 \times 10^{-12}$ for every dimensions.

Call this solver with:

    Gauss_Newton.Gauss_Newton(stations, event_test, n_iteration, steps, vp)

Where:
- `stations` are the normalised stations data with rows beeing the stations and the columns `X`, `Y`, `Z` positions and `t` time of detection difference compared to the reference station.
- `event_test` is the initial set of parameters (`X`, `Y`, `Z` positions and `t` the time of the failure relative to the reference station) from which the solver will search.
- `n_iteration` is the maximum number of iteration the solver can do.
- `steps` are per parameter scale use by the solver with the dampling method to adapt the size of the steps and help convergence.
- `vp` is the P-wave velocity in the world box and is assumed to be constant.

#### Levenberg-Marquardt:
This solver use Levenberg-Marquardt method to search for the earthquake hypocenter and failure timing.

The dampling method is completed by a reset control to keep the search in the ground (Z <= 0) same as the Gauss-Newton solver.

An early stoping is raised when the steps are smaller than $1 \times 10^{-12}$ for every dimensions.

Call this solver with:

    Levenberg_Marquardt.Levenberg_Marquardt(stations, event_test, n_iteration, vp)

Where:
- `stations` are the normalised stations data with rows beeing the stations and the columns `X`, `Y`, `Z` positions and `t` time of detection difference compared to the reference station.
- `event_test` is the initial set of parameters (`X`, `Y`, `Z` positions and `t` the time of the failure relative to the reference station) from which the solver will search.
- `n_iteration` is the maximum number of iteration the solver can do.
- `vp` is the P-wave velocity in the world box and is assumed to be constant.


#### Unique searcher:
This solver iterate by small steps toward the solution. At each iteration for every dimension (X, Y, Z and t), it test if increasing its position lead to lowest RMSE. If it is not the case the position is decrease. 

It incorporate a smoothing "learning rate" like factor to decrease the length of the steps when the misfit increased after he moved in all four dimensions. It also is constraint to not cross Z > 0.

An early stoping is raised when RMSE is lower than $1 \times 10^{-8}$ or when the decreasing rate is to slow.

Call this solver with:

    Uniq_searcher.gradient_descent(stations, event_test, n_iteration, steps, lr_decay, patience, vp)

Where:
- `stations` are the normalised stations data with rows beeing the stations and the columns `X`, `Y`, `Z` positions and `t` time of detection difference compared to the reference station.
- `event_test` is the initial set of parameters (`X`, `Y`, `Z` positions and `t` the time of the failure relative to the reference station) from which the solver will search.
- `n_iteration` is the maximum number of iteration the solver can do.
- `steps` are per parameter specified learning rate and is modified during the run by `lr_decay` to adapt the size of the steps.
- `lr_decay` is a general learning rate decaying, transformed in learning rate to update `steps` and must be in the interval [0, 1[.
- `patience` is used to detect if the misfit has stoped to decrease to trigger early stopping.
- `vp` is the P-wave velocity in the world box and is assumed to be constant.

#### Deepening grid:
The idea of this solver is to split the dimension into samples and to zoom iteratively toward the lowest RMSE. Sampling is carried out in such a way that the samples have all possible combinaison. With the four dimensions, ten split will give: $10^4$ samples.

This solver have two possible mode:
1. grid search: tested samples are the splited box center.
2. grid sampling: tested samples are randomly sampled in each splited box.

In both case, the recorded best event and linked RMSE will came from box centers around which the zoom will occur.

Call this solver with:

    deepening_grid.deepening_grid_search(stations, sampling_f, limites, depth, halving_rate, vp)

    # or

    deepening_grid.deepening_grid_sampling(stations, sampling_f, limites, depth, halving_rate, n_samples, vp)

Where:
- `stations` are the normalised stations data with rows beeing the stations and the columns `X`, `Y`, `Z` positions and `t` time of detection difference compared to the reference station.
- `sampling_f` is the number of sampling done per dimension and around which the solver will focus after iteration. The number of space split is computed by: $`\text{sampling\_f}^{\text{n dimensions}}`$.
- `limites` are the upper and lower bounds of the search space within the samples will be created and tested.
- `depth` is the maximum number of iteration (box splitting) the solver can do.
- `halving_rate` is (in percentage) the speed at which the search space is reduced and must be between 0.0 (no zoom at all) and 1.0 (the space is shrinked to a single sample in one iteration).
- `n_samples` is the number of samples generated during the misfit estimation of the sub-box. It will scale the total number of generated and evaluated samples to: $`\left( \text{n\_samples} + 1 \right) \times \text{sampling\_f}^{\text{n dimensions}}`$.
- `vp` is the P-wave velocity in the world box and is assumed to be constant.


#### Monte-Carlo one pass:
This solver compute a large set of random samples and score them.

It does not make any iteration but only one pass.

It can sample the defined search space with two methods:
1. Uniform: sample each dimension uniformly,
2. Perturbated grid: sample with equal distance every dimention with full combinaison. Total number of samples can be lower to asked to respect the equal number of sampling per dimension.
3. Face-centred cubic: a tenth of the requested samples are drawn in the `X`, `Y` and `Z` space with face-centred cubic meshing and perturbated, then, each of them are duplicated ten times with uniformly sampled `t` dimension.

Call this solver with:

    monte_carlo.monte_carlo(stations, limites, n_samples, sampling, vp)

Where:
- `stations` are the normalised stations data with rows beeing the stations and the columns `X`, `Y`, `Z` positions and `t` time of detection difference compared to the reference station.
- `limites` are the upper and lower bounds of the search space within the samples will be created and tested.
- `n_samples` is the number of samples to generate and evaluate. Some method will not actually compute the exact number of asked samples but more or less.
- `sampling` is the ranmdom sampling method, there are 'uniform' (1.), 'grid_rand' (2.) and 'fcc' (3.).
- `vp` is the P-wave velocity in the world box and is assumed to be constant.


#### Evolutionary:
This solver uses a evolutinary algorithm to search for the earthquake hypocenter and origin time. This implementation use elitist selection strategy. The population is reconstructed using: survivors of the previous generation, cross-over between all survivors, 3 mutating childrens per survivors and new random samples are used to keep population size stable.

Cross-over are done with full combinaison per couples to avoid values losses.

The mutated childs use CMA-ES (Covariance Matrix Adaptation Evolution Strategy) with Cholesky decomposition. Covaraince matrix is kept positive and defined with the diagonal being $max(value^i_i, 1\times 10^{-16})$.

The samples are constrained to stay below the air (Z $\leq$ 0) through a penalty: $score = score \times 10$ if $Z > 0$.

An early stoping is raised when RMSE is lower than $1 \times 10^{-8}$ or when the decreasing rate is to slow.

Call this solver with:

    Evolutionary.evolution(stations, limites, n_iteration, pop_size, p_surv, patience, vp)

Where:
- `stations` are the normalised stations data with rows beeing the stations and the columns `X`, `Y`, `Z` positions and `t` time of detection difference compared to the reference station.
- `limites` are the upper and lower bounds of the search space within the samples will be created and tested.
- `n_iteration` is the maximum number of iteration (number of generation here) the solver can do.
- `pop_size` is the total number of samples to evaluate at each generation.
- `p_surv` is the proportion of the samples which will survive and be used to reconstruct the population for the next generation.
- `patience` is used to detect if the misfit has stoped to decrease to trigger early stopping.
- `vp` is the P-wave velocity in the world box and is assumed to be constant.


### Solver to add in order:
1. Particule Swarm optimization solver;
