
"""
Script for Monte-Calro.
- meshgrid_numba_flat_3d:
    Creates flattened 4D grid coordinates using Numba.

- calc_misfit_numba:
    Compute RMSE misfit between expected and one observed arrival time.

- compute_loss_all:
    Loop on all `samples` to compute their RMSE misfit between expected and
    observed arrival time and create cost and history.

- monte_carlo:
    Function to compute a Monte Carlo method with one pass sampling.

"""

import math
import numpy as np
from numba import jit, njit, prange

@jit(nopython=True)
def meshgrid_numba_flat_3d(x:np.ndarray, y:np.ndarray, z:np.ndarray,
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

def monte_carlo(n_samples:int, limites:np.ndarray, stations:np.ndarray,
                sampling:str='uniform', vp:float=4000.0) -> tuple:
    """
    Function to compute a Monte Carlo method with one pass sampling.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    limites : numpy.ndarray
        A 2d array listing the limites of the search. The shape is (n, 2).
        n is the number of features. The n-th line is: [lower, upper] bound.
    stations : numpy.ndarray
        Station locations and picked arrival time.
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
        samples = np.random.uniform(0, 1, (n_samples, 4)).astype(np.float32)
        samples = samples*(lim_f32[:, 1]-lim_f32[:, 0])+lim_f32[:, 0]

    elif sampling == 'grid_rand':
        sub_n = int(n_samples**(1/4))
        differ = ((lim_f32[:, 1]-lim_f32[:, 0])/sub_n)/2
        samples = meshgrid_numba_flat_3d(
            np.linspace(lim_f32[0, 0], lim_f32[0, 1], sub_n),
            np.linspace(lim_f32[1, 0], lim_f32[1, 1], sub_n),
            np.linspace(lim_f32[2, 0], lim_f32[2, 1], sub_n),
            np.linspace(lim_f32[3, 0], lim_f32[3, 1], sub_n))

        noise = np.random.uniform(-1, 1, samples.shape
                                  ).astype(np.float32)*differ[None]

        samples = samples+noise

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
