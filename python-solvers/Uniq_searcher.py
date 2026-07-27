
"""
Script for unique searcher solver.
- calc_misfit_numba:
    Calculates the Root Mean Square (RMS) travel-time misfit for a seismic
    event location hypothesis.

- descent_loop:
    Executes a coordinate-descent search loop to optimize seismic event
    hypocenter location and origin time.

- gradient_descent
    Function to make a gradient descent from a given test event and data from
    stations

"""

import math
import numpy as np
from numba import njit

@njit(fastmath=True)
def calc_misfit_numba(stations_arr:np.ndarray, event_arr:np.ndarray,
                      vp:float=4000.0) -> float:
    """
    Calculates the Root Mean Square (RMS) travel-time misfit for a seismic
    event location hypothesis.

    Parameters
    ----------
    stations_arr : numpy.ndarray
        Array of shape (N, 4) containing seismic station coordinates and
        observed arrival times, where each row represents [X, Y, Z, t_obs].
    event_arr : numpy.ndarray
        Array of shape (4,) representing the hypocenter location and origin
        time hypothesis: [X, Y, Z, t_est].
    vp : float, optional
        Assumed constant P-wave propagation velocity in meters/second (or
        matching spatial units/s). The default is 4000.0.

    Returns
    -------
    float
        The Root Mean Squared Error arrival-time (misfit) across all N
        stations.

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

@njit(fastmath=True)
def descent_loop(stations_arr:np.ndarray, event_arr:np.ndarray,
                 n_iteration:int, steps:np.ndarray, patience:int,
                 lr_upd:float, vp:float, p3p:int, p2p:int) -> tuple:
    """
    Executes a coordinate-descent search loop to optimize seismic event
    hypocenter location and origin time.

    Performs axis-by-axis spatial and temporal perturbations (`X, Y, Z, t`).
    If no reduction  in misfit is achieved, step sizes are decayed. Includes
    spatial depth constraints (maintains Z <= 0)  and an early stopping
    mechanism based on cost flattening across historical patience intervals.

    Parameters
    ----------
    stations_arr : numpy.ndarray
        Array of shape (N, 4) with station coordinates and observed arrival
        times `[X, Y, Z, t_obs]`.
    event_arr : numpy.ndarray
        Initial event state array of shape (4,) representing `[X, Y, Z, t]`.
    n_iteration : int
        Maximum number of optimization iterations to run.
    steps : numpy.ndarray
        Array of shape (4,) containing initial step sizes/perturbations for
        each dimension `[dX, dY, dZ, dt]`.
    patience : int
        Number of iterations to evaluate for early stopping criteria.
    lr_upd : float
        Step decay multiplier applied when an iteration fails to lower misfit
        (typically `1 - lr_decay`).
    vp : float
        Assumed constant P-wave velocity.
    p3p : int
        Iteration threshold multiplier for enabling early stopping evaluation
        (typically `patience * 3`).
    p2p : int
        Historical lookback offset for relative misfit comparison (typically
        `patience * 2`).

    Returns
    -------
    history : numpy.ndarray
        Array of shape (completed_iterations + 1, 4) tracking the event
        parameters `[X, Y, Z, t]` at each step.
    cost_story : numpy.ndarray
        Array of shape (completed_iterations + 1,) recording the RMSE between
        expected and computed arrival times.
    i : int
        The index of the last completed iteration before termination or early
        stopping.

    """
    lr_h = 1

    history = np.empty((n_iteration+1, 4))
    history[0] = event_arr

    cost_story = np.empty(n_iteration+1)
    cost_story[0] = calc_misfit_numba(stations_arr, event_arr, vp)
    for i in range(n_iteration):
        current_misfit = cost_story[i]
        rec_mis = cost_story[i]
        for dim in range(4):
            event_arr[dim] += steps[dim]
            test_misfit = calc_misfit_numba(stations_arr, event_arr, vp)
            if test_misfit < current_misfit:
                current_misfit = test_misfit
                continue

            event_arr[dim] -= 2*steps[dim]
            test_misfit = calc_misfit_numba(stations_arr, event_arr, vp)
            current_misfit = test_misfit

        # stop going Z > 0
        if event_arr[2] > 0:
            event_arr[2] -= 2*steps[2]
            test_misfit = calc_misfit_numba(stations_arr, event_arr, vp)
            current_misfit = test_misfit

        # decay[0.5, 0.008] -> 0.01
        if rec_mis <= current_misfit:
            steps *= lr_upd
            lr_h  *= lr_upd

        history[i+1] = event_arr
        cost_story[i+1] = current_misfit
        if (i > p3p)&(lr_h < 0.01):
            if (i%patience) == 0:
                # Early stoping
                min_1 = np.min(cost_story[i-patience:i+1])
                if min_1 < 1e-8:
                    break

                min_2 = np.min(cost_story[i-p2p:i-patience])
                if min_2 < 1e-8:
                    break

                if (1-min_1/min_2) <= 1e-6:
                    break

    return history, cost_story, i

def gradient_descent(stations:np.ndarray, event_test:np.ndarray,
                     n_iteration:int, ranX:float=1, ranY:float=1,
                     ranZ:float=1, rant:float=0.0005, lr_decay:float=0.05,
                     patience:int=500, vp:float=4000.0):
    """
    Function to make a gradient descent from a given test event and data from
    stations.

    Parameters
    ----------
    stations : numpy.ndarray
        Station locations and picked arrival time.
    event_test : numpy.ndarray
        Event parameters : location and time {X, T, Z, t}.
    n_iteration : int
        Maximal number of iteration.
    ranX : float, optional
        Specified learning rate for the X parameter. The default is 1.0.
    ranY : float, optional
        Specified learning rate for the Y parameter. The default is 1.0.
    ranZ : float, optional
        Specified learning rate for the Z parameter. The default is 1.0.
    rant : float, optional
        Specified learning rate for the t parameter. The default is 0.0005.
    lr_decay : float, optional
        General learning rate decaying. Must be in [0, 1[. The default is
        0.05.
    patience : int, optional
        Patience for the early stoping. The default is 1000.
    vp : float, optional
        Assumed constant P-wave velocity. The default is 4000.0.

    Returns
    -------
    event_test : dict
        Best event founded with the lower root mean square error.
    cost_story : numpy.ndarray
        Loss history at each iteration.
    history : numpy.ndarray
        Listing of the best event at each iteration.

    """
    # Specified learning rate
    steps = np.array([ranX, ranY, ranZ, rant])

    lr_upd = 1-lr_decay

    p3p = int(patience*3)
    p2p = int(patience*2)

    # Gives us as an indication the misfit before applying gradient descent
    history, cost_story, index = descent_loop(
        stations, event_test, n_iteration, steps, patience, lr_upd, vp, p3p,
        p2p)

    history = history[:index+2]
    cost_story = cost_story[:index+2]
    best_idx = np.argmin(cost_story)
    misfit_fin = float(cost_story[best_idx])
    event_test = {'X':float(history[best_idx, 0]),
                  'Y':float(history[best_idx, 1]),
                  'Z':float(history[best_idx, 2]),
                  't':float(history[best_idx, 3])}

    return event_test, history, cost_story, misfit_fin
