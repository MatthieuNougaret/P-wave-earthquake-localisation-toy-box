
"""
Script for deepening grid solver with and without sub sampling.

- meshgrid_numba_flat_4d:
    Creates flattened 4D grid coordinates using Numba.

- calc_misfit_numba:
    Compute RMSE misfit between expected and one observed arrival time.

- compute_loss_all:
    Loop on all `samples` to compute their RMSE misfit between expected and
    observed arrival time.

- deepen_grid_loop:
    Main iterative loop where the deepening happen.

- deepening_grid_search:
    Function to be call to compute the deepening grid search and what numba
    can not work with.

- compute_per_sampling:
    Function to estimate level of interest of the box with random sub
    sampling and dynamic Softplus averaging.

- deepen_grid_loop_sampling:
    Main iterative loop where the deepening happen.

- deepening_grid_sampling:
    Function to be call to compute the deepening grid with sub sampling
    search and what numba can not work with.

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
    # Standard 'ij indexing' matching meshgrid default order (z, y, x, t)
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
def compute_loss_all(samples_arr:np.ndarray, stations_arr:np.ndarray,
                     vp:float=4000.0) -> np.ndarray:
    """
    Loop on all `samples` to compute their RMSE misfit between expected and
    observed arrival time.

    Parameters
    ----------
    samples_arr : numpy.ndarray
        Samples from which the function will compute their RMSE misfit
        between expected and observed arrival time. Shape (M, 4) -> [[X_i,
        Y_i, Z_i, t_tes_i]... ].
    stations_arr : numpy.ndarray
        2d array, normalised stations parameters. Shape (N, 4) -> [[X_i, Y_i,
        Z_i, t_tes_i]... ].
    vp : float, optional
        Assumed constant P-wave velocity. The default is 4000.0.

    Returns
    -------
    loss : numpy.ndarray
        Vector with the RMSE misfit of the given samples.

    """
    n_samples = len(samples_arr)
    loss = np.empty(shape=(n_samples,))
    for i in range(n_samples):
        loss[i] = calc_misfit_numba(stations_arr, samples_arr[i], vp)

    return loss

@njit(fastmath=True)
def deepen_grid_loop(stations:np.ndarray, sampling_f:float,
                     limites:np.ndarray, depth:int, halving_rate:float=0.2,
                     vp:float=4000.0) -> tuple:
    """
    Main iterative loop where the deepening happen.
    It fill `cost_story` with lowest RMSE found trhough the candidate and
    `history` with the lowest RMSE candidate parameters.

    Parameters
    ----------
    stations : numpy.ndarray
        2d array, normalised stations parameters. Shape (N, 4) -> [X, Y, Z,
        t_obs].
    sampling_f : int
        Sampling frequence of the space of parameters to create samples. The
        number of samples will be sampling_f to the power four.
    limites : numpy.ndarray
        A 2d array listing the limites of the search. The shape is (n, 2).
        n is the number of features. The n-th line is: [lower, upper] bound.
    depth : int
        Depth of the search, i.e. the number of epoch.
    halving_rate : float
        Define the speed at which the search space is reduced. Must be:
        0 > hr > 1. This value indicate the reduction proportion of the
        search space at each step. With 0.20, it means: every iteration
        search on a 20% smaller space (10% of each side).
    vp : float, optional
        Assumed constant P-wave velocity. The default is 4000.0.

    Returns
    -------
    history : numpy.ndarray
        Sample created and tested. Shape is (depth, 4), its length may be
        reduce with `i` in `deepening_grid_search` function if early stopping
        was triggered.
    cost_hist : numpy.ndarray
        Root mean square error history of the best sample at each iteration.
        Shape is (depth, 4), its length may be reduce with `i` in
        `deepening_grid_search` function if early stopping was triggered.
    i : int
        Iteration at which the deepening stoped (aka the depth reached). It
        will be used in `deepening_grid_search` function if early stopping
        was triggered.

    """
    cost_story = np.empty(depth)
    history = np.empty((depth, 4))
    lims = limites.copy()
    for i in range(depth):
        diff = (lims[:, 1]-lims[:, 0])/2 *(1-halving_rate)
        delta = np.column_stack((-diff, diff))

        samples = meshgrid_numba_flat_4d(
            np.linspace(lims[0, 0], lims[0, 1], sampling_f),
            np.linspace(lims[1, 0], lims[1, 1], sampling_f),
            np.linspace(lims[2, 0], lims[2, 1], sampling_f),
            np.linspace(lims[3, 0], lims[3, 1], sampling_f))

        loss = compute_loss_all(samples, stations, vp)

        min_i = np.argmin(loss)
        current = samples[min_i]
        lims = current[:, None]+delta
        history[i] = current
        cost_story[i] = loss[min_i]

        # if the x, y and z limits are at sub cm search
        if np.all((lims[:-1, 1]-lims[:-1, 0]) < 1e-3):
            break

        elif cost_story[i] < 1e-8:
            break

        if lims[2, 1] > 0:
            lims[2] = lims[2]-lims[2, 1]+(lims[2, 1]-lims[2, 0])/(sampling_f+1)

    return history, cost_story, i

def deepening_grid_search(stations:np.ndarray, sampling_f:float,
                          limites:np.ndarray, depth:int,
                          halving_rate:float=0.20, vp:float=4000.0) -> tuple:
    """
    Function to be call to compute the deepening grid search and what numba
    can not work with.

    Parameters
    ----------
    stations : numpy.ndarray
        2d array, normalised stations parameters. Shape (N, 4) -> [X, Y, Z,
        t_obs].
    sampling_f : int
        Sampling frequence of the space of parameters to create samples. The
        number of samples will be sampling_f to the power four.
    limites : numpy.ndarray
        A 2d array listing the limites of the search. The shape is (n, 2).
        n is the number of features. The n-th line is: [lower, upper] bound.
    depth : int
        Depth of the search, i.e. the number of epoch.
    halving_rate : float, optional
        Define the speed at which the search space is reduced. Must be:
        0 > hr > 1. This value indicate the reduction proportion of the
        search space at each step. With 0.20, it means: every iteration
        search on a 20% smaller space (10% of each side). The default is
        0.20.
    vp : float, optional
        Assumed constant P-wave velocity. The default is 4000.0.

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
    history, cost_story, index = deepen_grid_loop(stations, sampling_f,
                                                  limites, depth,
                                                  halving_rate, vp)

    history = history[:index+1]
    cost_story = cost_story[:index+1]
    best_idx = np.argmin(cost_story)
    misfit_fin = float(cost_story[best_idx])
    event_test = {'X':float(history[best_idx, 0]),
                  'Y':float(history[best_idx, 1]),
                  'Z':float(history[best_idx, 2]),
                  't':float(history[best_idx, 3])}

    return event_test, history, cost_story, misfit_fin

@njit(fastmath=True)
def compute_per_sampling(samples:np.ndarray, n_samples:int,
                         stations:np.ndarray, box_prop:np.ndarray,
                         vp:float=4000.0) -> tuple:
    """
    Function to estimate level of interest of the box with random sub
    sampling and dynamic Softplus averaging.

    Parameters
    ----------
    samples_arr : numpy.ndarray
        Samples are the center of the box to estimate (from meshgrid). Shape
        (M, 4) -> [[X_i, Y_i, Z_i, t_tes_i]... ].
    n_samples : int
        Number of random samples to draw to estimate each box.
    stations : numpy.ndarray
        2d array, normalised stations parameters. Shape (N, 4) -> [X, Y, Z,
        t_obs].
    box_prop:np.ndarray
        Vector with half of the box size to make random sampling using (-1,
        1) bounds.
    vp : float, optional
        Assumed constant P-wave velocity. The default is 4000.0.

    Returns
    -------
    sampling_loss : numpy.ndarray
        Vector with Softplus averaged RMSE misfit of the sub sampling.
    loss : numpy.ndarray
        Vector with RMSE misfit between given samples (grid box centers of
        the meshgrid).

    """
    beta = 1
    loss = compute_loss_all(samples, stations, vp)
    sampling_loss = np.empty(len(samples))
    for i in range(len(samples)):
        noise = np.random.uniform(-1, 1, size=(n_samples, 4))
        neig = samples[i] + noise * box_prop
        loss_i = compute_loss_all(neig, stations, vp)
        std_val = np.std(loss_i)
        beta = 2.0 / std_val if std_val > 1e-8 else 1.0
        min_loss = np.min(loss_i)
        shift_loss = loss_i - min_loss

        # Avoids underflow / overflow in exp()
        weights = np.exp(-shift_loss * beta)
        sampling_loss[i] = min_loss - (1.0 / beta) * np.log(np.mean(weights))

    return sampling_loss, loss

@njit(fastmath=True)
def deepen_grid_loop_sampling(stations:np.ndarray, sampling_f:float,
                              limites:np.ndarray, depth:int,
                              halving_rate:float=0.2, n_samples:int=100,
                              vp:float=4000.0) -> tuple:
    """
    Main iterative loop where the deepening happen.
    It fill `cost_story` with the box center which have the lowest RMSE
    estimation trhough subsampling. `history` get the parameters of the
    selected box center.

    Parameters
    ----------
    stations : numpy.ndarray
        2d array, normalised stations parameters. Shape (N, 4) -> [X, Y, Z,
        t_obs].
    sampling_f : int
        Sampling frequence of the space of parameters to create samples. The
        number of samples will be sampling_f to the power four.
    limites : numpy.ndarray
        A 2d array listing the limites of the search. The shape is (n, 2).
        n is the number of features. The n-th line is: [lower, upper] bound.
    depth : int
        Depth of the search, i.e. the number of epoch.
    halving_rate : float
        Define the speed at which the search space is reduced. Must be:
        0 > hr > 1. This value indicate the reduction proportion of the
        search space at each step. With 0.20, it means: every iteration
        search on a 20% smaller space (10% of each side).
    n_samples : int, optional
        Number of samples generated during box misfit estimation. The default
        is 100.
    vp : float, optional
        Assumed constant P-wave velocity. The default is 4000.0.

    Returns
    -------
    history : numpy.ndarray
        Sample created and tested. Shape is (depth, 4), its length may be
        reduce with `i` in `deepening_grid_search` function if early stopping
        was triggered.
    cost_hist : numpy.ndarray
        Root mean square error history of the best sample at each iteration.
        Shape is (depth, 4), its length may be reduce with `i` in
        `deepening_grid_search` function if early stopping was triggered.
    i : int
        Iteration at which the deepening stoped (aka the depth reached). It
        will be used in `deepening_grid_search` function if early stopping
        was triggered.

    """
    cost_story = np.empty(depth)
    history = np.empty((depth, 4))
    lims = limites.copy()
    for i in range(depth):
        diff = (lims[:, 1]-lims[:, 0]) / 2
        box_prop = diff / (sampling_f-1)

        samples = meshgrid_numba_flat_4d(
            np.linspace(lims[0, 0], lims[0, 1], sampling_f),
            np.linspace(lims[1, 0], lims[1, 1], sampling_f),
            np.linspace(lims[2, 0], lims[2, 1], sampling_f),
            np.linspace(lims[3, 0], lims[3, 1], sampling_f))

        samploss, loss  = compute_per_sampling(samples, n_samples, stations,
                                               box_prop, vp)

        min_i = np.argmin(samploss)
        current = samples[min_i]
        history[i] = current
        cost_story[i] = loss[min_i]

        # if the x, y and z limits are at sub cm search
        if np.all((lims[:-1, 1]-lims[:-1, 0]) < 1e-3):
            break

        elif cost_story[i] < 1e-8:
            break

        diff = diff * (1-halving_rate)
        delta = np.column_stack((-diff, diff))
        lims = current[:, None]+delta

        if lims[2, 1] > 0:
            lims[2] = lims[2]-lims[2, 1]+(lims[2, 1]-lims[2, 0])/(sampling_f+1)

    return history, cost_story, i

def deepening_grid_sampling(stations:np.ndarray, sampling_f:float,
                            limites:np.ndarray, depth:int,
                            halving_rate:float=0.20, n_samples:int=100,
                            vp:float=4000.0) -> tuple:
    """
    Function to be call to compute the deepening grid with sub sampling
    search and what numba can not work with.

    Parameters
    ----------
    stations : numpy.ndarray
        2d array, normalised stations parameters. Shape (N, 4) -> [X, Y, Z,
        t_obs].
    sampling_f : int
        Sampling frequence of the space of parameters to create samples. The
        number of samples will be sampling_f to the power four.
    limites : numpy.ndarray
        A 2d array listing the limites of the search. The shape is (n, 2).
        n is the number of features. The n-th line is: [lower, upper] bound.
    depth : int
        Depth of the search, i.e. the number of epoch.
    halving_rate : float, optional
        Define the speed at which the search space is reduced. Must be:
        0 > hr > 1. This value indicate the reduction proportion of the
        search space at each step. With 0.20, it means: every iteration
        search on a 20% smaller space (10% of each side). The default is
        0.20.
    n_samples : int, optional
        Number of samples generated during box misfit estimation. The default
        is 100.
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
    history, cost_story, index = deepen_grid_loop_sampling(
        stations, sampling_f, limites, depth, halving_rate, n_samples, vp)

    history = history[:index+1]
    cost_story = cost_story[:index+1]
    best_idx = np.argmin(cost_story)
    misfit_fin = float(cost_story[best_idx])
    event_test = {'X':float(history[best_idx, 0]),
                  'Y':float(history[best_idx, 1]),
                  'Z':float(history[best_idx, 2]),
                  't':float(history[best_idx, 3])}

    return event_test, history, cost_story, misfit_fin
