
"""
Script for Monte-Calro.
- meshgrid_numba_flat_4d:
    Creates flattened 4D grid coordinates using Numba.

- calc_misfit_numba:
    Compute RMSE misfit between expected and one observed arrival time.

- compute_loss_all:
    Loop on all `samples` to compute their RMSE misfit between expected and
    observed arrival time and create cost and history.

- uniform_sampling:
    Draw `n_samples` candidate events uniformly at random within the given
    per-dimension bounds.

- perturbated_grid_sampling:
    Draw candidate events on a regular 4D grid over the search space, then
    jitter each point with uniform noise within its own grid cell.

- fcc_lattice_sampling:
    Draw candidate spatial (X, Y, Z) points laid out on a jittered face-
    centred-cubic lattice, then jitter each point with uniform noise.

- fcc_sampling:
    Create the samples to test by first sampling X, Y and Z space then
    duplicating the positions to samples multiple relative failure time per
    hypocenters

- monte_carlo:
    Function to compute a Monte Carlo method with one pass sampling.

"""

import math
import numpy as np
from numba import jit, njit, prange

@jit(nopython=True)
def meshgrid_numba_flat_4d(x:np.ndarray, y:np.ndarray, z:np.ndarray,
                           t:np.ndarray) -> np.ndarray:
    """
    Creates flattened 4D grid coordinates using Numba.

    Parameters
    ----------
    x : numpy.ndarray
        Vector with the x values to combine.
    y : numpy.ndarray
        Vector with the y values to combine.
    z : numpy.ndarray
        Vector with the z values to combine.
    t : numpy.ndarray
        Vector with the t values to combine.

    Returns
    -------
    xyzt : numpy.ndarray
        2d array of x, y, z and t event candidate from the hypergrid. Shape
        is (N, 4) -> [[X_i, Y_i, Z_i, t_i], ...].

    """
    n_x = x.size
    total_size = n_x * n_x * n_x * n_x
    xyzt = np.empty(shape=(total_size, 4), dtype=np.float32)
    idx = 0
    for i in range(n_x):
        for j in range(n_x):
            for k in range(n_x):
                for l in range(n_x):
                    xyzt[idx, 0] = x[i]
                    xyzt[idx, 1] = y[j]
                    xyzt[idx, 2] = z[k]
                    xyzt[idx, 3] = t[l]

                    idx += 1

    return xyzt

@njit(fastmath=True, inline="always")
def calc_misfit_numba(stations_arr:np.ndarray, event_arr:np.ndarray,
                      vp:float=4000.0) -> float:
    """
    Compute RMSE misfit between expected and one observed arrival time.

    Parameters
    ----------
    stations_arr : numpy.ndarray
        2d array, normalised stations parameters. Shape (N, 4) -> [X, Y, Z,
        t_obs].
    event_arr : numpy.ndarray
        1d array, event candidate parameters. Shape (4,) -> [X, Y, Z, t_est].
    vp : float, optional
        Assumed constant P-wave velocity. The default is 4000.0.

    Returns
    -------
    float
        RMSE between expected and observed arrival time.

    """
    n = stations_arr.shape[0]
    sum_sq = 0.0
    ex, ey, ez, et = event_arr[0], event_arr[1], event_arr[2], event_arr[3]
    for i in range(n):
        dx = stations_arr[i, 0] - ex
        dy = stations_arr[i, 1] - ey
        dz = stations_arr[i, 2] - ez

        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        tp_est = et + dist / vp
        err = tp_est - stations_arr[i, 3]
        sum_sq += err * err

    return math.sqrt(sum_sq / n)

@jit(nopython=True, fastmath=True)
def compute_loss_all(samples_f32:np.ndarray, stations_f32:np.ndarray,
                     vp:float=4000.0) -> np.ndarray:
    """
    Loop on all `samples_f32` to compute their RMSE misfit between expected and
    observed arrival time and create cost and history.

    Parameters
    ----------
    samples_f32 : numpy.ndarray
        Samples from which the function will compute their RMSE misfit
        between expected and observed arrival time. Shape (M, 4) -> [[X_i,
        Y_i, Z_i, t_tes_i]... ].
    stations_f32 : numpy.ndarray
        2d array, normalised stations parameters. Shape (N, 4) -> [[X_i, Y_i,
        Z_i, t_tes_i]... ].
    vp : float, optional
        Assumed constant P-wave velocity. The default is 4000.0.

    Returns
    -------
    history : numpy.ndarray
        Sample created and tested. Shape is (depth, 4), it will be cut with
        `c` in `monte_carlo` function.
    cost_hist : numpy.ndarray
        Root mean square error history of the best sample at each iteration.
        Shape is (depth, 4), it will be cut with `c` in `monte_carlo`
        function.
    c : int
        Iteration at which the deepening stoped (aka the depth reached). It
        will be used in `monte_carlo` function to cut history and cost_hist.

    """
    n_samples = len(samples_f32)
    cost_story = np.empty(shape=(n_samples, ), dtype=np.float32)
    history = np.empty(shape=(n_samples, 4), dtype=np.float32)
    c = 0
    cost_story[c] = calc_misfit_numba(stations_f32, samples_f32[c], vp)
    history[c] = samples_f32[c]
    for i in prange(1, n_samples, 1):
        loss_i = calc_misfit_numba(stations_f32, samples_f32[i], vp)
        if loss_i < cost_story[c]:
            cost_story[c+1] = loss_i
            history[c+1] = samples_f32[i]
            c += 1

    return history, cost_story, c

@njit(fastmath=True)
def uniform_sampling(n_samples:int, lim_f32:np.ndarray) -> np.ndarray:
    """
    Draw `n_samples` candidate events uniformly at random within the given
    per-dimension bounds.

    Parameters
    ----------
    n_samples : int
        Number of samples to draw.
    lim_f32 : numpy.ndarray
        Shape (4, 2) array of (lower, upper) bounds for X, Y, Z, t.

    Returns
    -------
    samples : numpy.ndarray
        Shape (n_samples, 4) array of sampled [X, Y, Z, t] points.

    """
    samples = np.random.uniform(0, 1, size=(n_samples, 4)).astype(np.float32)
    samples = samples*(lim_f32[:, 1]-lim_f32[:, 0])+lim_f32[:, 0]
    return samples

def perturbated_grid_sampling(n_samples:int, lim_f32:np.ndarray
                              ) -> np.ndarray:
    """
    Draw candidate events on a regular 4D grid over the search space, then
    jitter each point with uniform noise within its own grid cell.

    The grid resolution per dimension is `sub_n = floor(n_samples**(1/4))`,
    so the returned sample count is `sub_n**4`, which is generally less
    than `n_samples` (since `n_samples` need not itself be a perfect 4th
    power).

    Parameters
    ----------
    n_samples : int
        Target number of samples; the actual count returned is
        `floor(n_samples**(1/4))**4`.
    lim_f32 : numpy.ndarray
        Shape (4, 2) array of (lower, upper) bounds for X, Y, Z, t.

    Returns
    -------
    samples : numpy.ndarray
        Shape (sub_n**4, 4) array of sampled [X, Y, Z, t] points.

    """
    sub_n = int(n_samples**(1/4))
    differ = ((lim_f32[:, 1]-lim_f32[:, 0])/sub_n)/2
    samples = meshgrid_numba_flat_4d(
        np.linspace(lim_f32[0, 0], lim_f32[0, 1], sub_n),
        np.linspace(lim_f32[1, 0], lim_f32[1, 1], sub_n),
        np.linspace(lim_f32[2, 0], lim_f32[2, 1], sub_n),
        np.linspace(lim_f32[3, 0], lim_f32[3, 1], sub_n))

    noise = np.random.uniform(-1, 1, size=samples.shape
                              ).astype(np.float32)*differ[None]

    samples = samples+noise

    return samples

def fcc_lattice_sampling(n_samples:int, bounds:np.ndarray,
                         jitter_frac:float=0.66) -> np.ndarray:
    """
    Draw candidate spatial (X, Y, Z) points laid out on a jittered face-
    centred-cubic (FCC) lattice, then jitter each point with uniform noise.

    Construction: FCC's conventional cubic unit cell of side `a` contains
    4 lattice points -- (0,0,0), (a/2,a/2,0), (a/2,0,a/2), (0,a/2,a/2) --
    tiled by integer steps of `a` along each axis (an axis-aligned,
    rectangular-box-friendly tiling. `a` is chosen from the target sample
    count and domain volume so that `4 * (Lx/a) * (Ly/a) * (Lz/a) ~=
    n_samples`. Nearest-neighbour spacing on the unjittered lattice is
    `a / sqrt(2)`.

    Parameters
    ----------
    n_samples : int
        Target number of samples. The actual count returned is close to
        but not necessarily exactly `n_samples` (a whole number of unit
        cells must fit in the domain).
    bounds : numpy.ndarray
        Shape (3, 2) array of per-dimension (lower, upper) bounds for
        X, Y, Z.
    jitter_frac : float, optional
        Fraction of the unjittered lattice's nearest-neighbour spacing
        (`a/sqrt(2)`) used as the total jitter amplitude per axis (applied
        as `+/- jitter_frac * nn_dist / 2`). This is a soft
        perturbation, not a guaranteed minimum spacing: with large enough
        `jitter_frac`, jittered points from different lattice sites
        can end up closer together than the unjittered spacing (or, in
        principle, than each other). The default is 0.66.
 
    Returns
    -------
    samples : numpy.ndarray
        Shape (n, 3) array of sampled [X, Y, Z] points, `n` close to but
        not necessarily exactly `n_samples`.
 
    """
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    extents = upper - lower
    volume = np.prod(extents)

    a = (4.0 * volume / n_samples) ** (1/3)
    n_cells = np.ceil(extents / a).astype(int) + 1

    basis = np.array([[0.0, 0.0, 0.0],
                      [0.5, 0.5, 0.0],
                      [0.5, 0.0, 0.5],
                      [0.0, 0.5, 0.5]]) * a

    i, j, k = np.meshgrid(np.arange(n_cells[0]), np.arange(n_cells[1]),
                          np.arange(n_cells[2]), indexing='ij')

    cell_origins = np.stack([i.ravel(), j.ravel(), k.ravel()],
                            axis=1).astype(np.float64) * a

    points = (cell_origins[:, None, :] + basis[None, :, :]
             ).reshape(-1, 3) + lower

    in_bounds = np.all((points >= lower) & (points <= upper), axis=1)
    points = points[in_bounds]

    nn_dist = a / np.sqrt(2)
    jitter_amp = jitter_frac * nn_dist / 2
    noise = np.random.uniform(-jitter_amp, jitter_amp, size=points.shape)
    points = points + noise
    points = np.clip(points, lower, upper)

    return points.astype(np.float32)

def fcc_sampling(n_samples:int, bounds:np.ndarray, jitter_frac:float=0.66):
    """
    Create the samples to test by first sampling X, Y and Z space then
    duplicating the positions to samples multiple relative failure time per
    hypocenters.

    Parameters
    ----------
    n_samples : int
        Total number of samples to create.
    bounds : numpy.ndarray
        Shape (4, 2) array of (lower, upper) bounds for X, Y, Z, t.
    jitter_frac : float, optional
        Fraction of the unjittered lattice's nearest-neighbour spacing
        (a/sqrt(2)) used as the total jitter amplitude per axis (applied
        as +/- jitter_frac *nn_dist /2). The default is 0.66.

    Returns
    -------
    samples : numpy.ndarray
        Shape (sub_n**4, 4) array of sampled [X, Y, Z, t] points.

    """
    n_space_samps = n_samples // 10
    space_bounds = np.copy(bounds[:3])
    positions = fcc_lattice_sampling(n_space_samps, space_bounds,
                                     jitter_frac)

    ngene = len(positions)
    samples = np.empty((ngene*10, 4))
    samples[:, :3] = np.repeat(positions, 10).reshape(ngene, 3, 10
                               ).transpose(0, 2, 1).reshape(ngene*10, 3)

    samples[:, 3] = np.random.uniform(bounds[3, 0], bounds[3, 1],
                                      size=samples.shape[0])

    return samples.astype(np.float32)

def monte_carlo(stations:np.ndarray, limites:np.ndarray, n_samples:int,
                sampling:str='uniform', vp:float=4000.0) -> tuple:
    """
    Function to compute a Monte Carlo method with one pass sampling.

    Parameters
    ----------
    stations : numpy.ndarray
        Station locations and picked arrival time.
    limites : numpy.ndarray
        A 2d array listing the limites of the search. The shape is (n, 2).
        n is the number of features. The n-th line is: [lower, upper] bound.
    n_samples : int
        Number of samples.
    sampling : str, optional
        Method to sampling the parameter space. If 'random', it will simply
        draw n_samples random values for each parameters. If 'grid_rand', it
        will first define a 4d grid with a total of cells <= n_samples. Then
        it will move randomly these samples within their cell. The default is
        'random'.
    vp : float, optional
        Assumed constant P-wave velocity. The default is 4000.

    Returns
    -------
    event_test : dictionary
        Best event candidate (lowest RMSE).
        {X:float, Y:float, Z:float, t:float}
    history : numpy.ndarray
        2d array with the history of the best event candidate.
    cost_hist : numpy.ndarray
        Root Mean Square Error history of the best sample at each iteration.
    misfit_fin : float
        RMSE of `event_test`, the lowest misfit found during the search.

    """
    lim_f32 = limites.astype(np.float32)
    stations_f32 = stations.astype(np.float32)
    if sampling == 'uniform':
        samples = uniform_sampling(n_samples, lim_f32)

    elif sampling == 'grid_rand':
        samples = perturbated_grid_sampling(n_samples, lim_f32)

    elif sampling == 'fcc':
        samples = fcc_sampling(n_samples, lim_f32, 0.75)

    history, cost_story, index = compute_loss_all(samples, stations_f32, vp)

    history = history[:index+1].astype(np.float64)
    cost_story = cost_story[:index+1].astype(np.float64)
    best_idx = np.argmin(cost_story)
    misfit_fin = float(cost_story[best_idx])
    event_test = {'X':float(history[best_idx, 0]),
                  'Y':float(history[best_idx, 1]),
                  'Z':float(history[best_idx, 2]),
                  't':float(history[best_idx, 3])}

    return event_test, history, cost_story, misfit_fin
