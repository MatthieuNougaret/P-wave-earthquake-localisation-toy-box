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

#### Levenberg-Marquardt:
This solver use Levenberg-Marquardt method to search for the earthquake hypocenter and failure timing.

The dampling method is completed by a reset control to keep the search in the ground (Z <= 0) same as the Gauss-Newton solver.

An early stoping is raised when the steps are smaller than $1 \times 10^{-12}$ for every dimensions.

#### Unique searcher:
This solver iterate by small steps toward the solution. At each iteration for every dimension (X, Y, Z and t), it test if increasing its position lead to lowest RMSE. If it is not the case the position is decrease. 

It incorporate a smoothing "learning rate" like factor to decrease the length of the steps when the misfit increased after he moved in all four dimensions. It also is constraint to not cross Z > 0.

An early stoping is raised when RMSE is lower than $1 \times 10^{-8}$ or when the decreasing rate is to slow.

#### Deepening grid:
The idea of this solver is to split the dimension into samples and to zoom iteratively toward the lowest RMSE. Sampling is carried out in such a way that the samples have all possible combinaison. With the four dimensions, ten split will give: $10^4$ samples.

This solver have two possible mode:
1. grid search: tested samples are the splited box center.
2. grid sampling: tested samples are randomly sampled in each splited box.

In both case, the recorded best event and linked RMSE will came from box centers around which the zoom will occur.

### Monte-Carlo one pass:
This solver compute a large set of random samples and score them.

It does not make any iteration but only one pass.

It can sample the defined search space with two methods:
1. Uniform: sample each dimension uniformly,
2. Perturbated grid: sample with equal distance every dimention with full combinaison. Total number of samples can be lower to asked to respect the equal number of sampling per dimension.

### Evolutionary:
This solver uses a evolutinary algorithm to search for the earthquake hypocenter and origin time. This implementation use elitist selection strategy. The population is reconstructed using: survivors of the previous generation, cross-over between all survivors, 3 mutating childrens per survivors and new random samples are used to keep population size stable.

Cross-over are done with full combinaison per couples to avoid values losses.

The mutated childs use CMA-ES (Covariance Matrix Adaptation Evolution Strategy) with Cholesky decomposition. Covaraince matrix is kept positive and defined with the diagonal being $max(value^i_i, 1\times 10^{-16})$.

The samples are constrained to stay below the air (Z $\leq$ 0) through a penalty: $score = score \times 10$ if $Z > 0$.

An early stoping is raised when RMSE is lower than $1 \times 10^{-8}$ or when the decreasing rate is to slow.

### Solver to add in order:
1. Monte-Carlo poisson disc sampling;
2. Particule Swarm optimization solver;
