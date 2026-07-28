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
