
"""
Script for Levenberg-Marquardt solver.
- compute_residuals:
    Compute the residuals and distances between the candidate and the
    stations.

- compute_Jacobian:
    Compute the Jacobian matrix.

- loop_Lev_Mar:
    Search the hypocenter with gradient descent

- Levenberg_Marquardt:
    Prepare the gradient descent and compute what would block numba.

"""

import math
import numpy as np
from numba import njit

@njit(fastmath=True)
def compute_residuals(stations_arr:np.ndarray, event_arr:np.ndarray,
                      vp:float) -> tuple:
    """
    Function to compute the residuals and distances between the candidate and
    the stations.

    Parameters
    ----------
    stations_arr : numpy.ndarray
        Position (X, Y and Z) and relative detection time.
        array([[X_i, Y_i, Z_i, t_i], ...])
    event_arr : numpy.ndarray
        Event candidate to evaluate.
        array([X, Y, Z, t])
    vp : float
        P-wave speed propagation in m/s.

    Returns
    -------
    res : numpy.ndarray
        Vector with the residuals per stations.
    dist : numpy.ndarray
        Vector of the distances between the event candidate and the stations.
        To be used in Jacobian matrix computation.

    """
    n = stations_arr.shape[0]
    res = np.empty(n)
    dist = np.empty(n)

    ex, ey, ez, et = event_arr[0], event_arr[1], event_arr[2], event_arr[3]
    for i in range(n):
        dx = stations_arr[i, 0] - ex
        dy = stations_arr[i, 1] - ey
        dz = stations_arr[i, 2] - ez

        d = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-12
        tp_est = et + d / vp

        dist[i] = d
        res[i] = tp_est - stations_arr[i, 3]

    return res, dist

@njit(fastmath=True)
def compute_Jacobian(stations_arr:np.ndarray, event_arr:np.ndarray,
                     distances:np.ndarray, vp:float) -> np.ndarray:
    """
    Function to compute the Jacobian matrix.

    Parameters
    ----------
    stations_arr : numpy.ndarray
        Position (X, Y and Z) and relative detection time.
        array([[X_i, Y_i, Z_i, t_i], ...])
    event_arr : numpy.ndarray
        Event candidate to evaluate.
        array([X, Y, Z, t])
    distances : numpy.ndarray
        Vector of the distances between the event candidate and the stations.
    vp : float
        P-wave speed propagation in m/s.

    Returns
    -------
    J : numpy.ndarray
        Jacobian matrix.

    """
    n = stations_arr.shape[0]
    J = np.empty((n, 4))

    norm = distances * vp
    ex, ey, ez, et = event_arr[0], event_arr[1], event_arr[2], event_arr[3]
    for i in range(n):
        J[i, 0] = (ex-stations_arr[i, 0]) / norm[i]
        J[i, 1] = (ey-stations_arr[i, 1]) / norm[i]
        J[i, 2] = (ez-stations_arr[i, 2]) / norm[i]
        J[i, 3] = 1.0

    return J

@njit(fastmath=True)
def loop_Lev_Mar(stations:np.ndarray, event_test:np.ndarray,
                 n_iteration:int, vp:float=4000.0) -> tuple:
    """
    Function to search the hypocenter with gradient descent.

    Parameters
    ----------
    stations : numpy.ndarray
        Position (X, Y and Z) and relative detection time.
        array([[X_i, Y_i, Z_i, t_i], ...])
    event_test : numpy.ndarray
        Initial event candidate.
        array([X, Y, Z, t])
    n_iteration : int
        Maximum number of iteration.
    vp : float, optional
        P-wave speed propagation in m/s. The default is 4000.0.

    Returns
    -------
    history : numpy.ndarray
        Vector with the history of the event candidate.
    cost_story : numpy.ndarray
        Vector with the history of the RMSE of the event candidate.
    i : int
        iteration at which the descent stopped.

    """
    n_stations = len(stations)
    factor = 10.0
    lambda_lm = 1e-3

    start_point = np.copy(event_test)
    depth_res = 1.0

    res, dist = compute_residuals(stations, event_test, vp)
    misfit_current = np.sum(res**2)/2

    history = np.empty((n_iteration+1, 4))
    history[0] = event_test

    cost_story = np.empty(n_iteration+1)
    cost_story[0] = math.sqrt(np.sum(res**2 /n_stations))

    for i in range(n_iteration):
        J = compute_Jacobian(stations, event_test, dist, vp)

        Hess = J.T@J
        grad_v = J.T@res
        Hess_LM = Hess.copy()
        for d in range(4):
            Hess_LM[d, d] += lambda_lm * Hess[d, d]

        delta_m = np.linalg.solve(Hess_LM, -grad_v)

        tested = event_test + delta_m
        if tested[2] > 0.0:
            depth_res *= 1.1
            event_test[2] = start_point[2] * depth_res
            if event_test[2] > -1:
                event_test[2] = -1

            res, dist = compute_residuals(stations, event_test, vp)
            misfit_current = np.sum(res**2)/2

            lambda_lm = 1e-3

            delta_m = np.ones(4)
            grad_v = np.ones(4)

        else:
            errors, _ = compute_residuals(stations, tested, vp)
            r_test = np.sum(errors**2)/2
            if r_test < misfit_current:
                event_test = tested.copy()
                misfit_current = r_test
                dist = _.copy()
                res = errors.copy()
                lambda_lm = max(lambda_lm / factor, 1e-8)
    
            else:
                lambda_lm = min(lambda_lm * factor, 1e8)

        history[i+1] = event_test
        cost_story[i+1] = math.sqrt(np.sum(res**2 /n_stations))
        if np.max(np.abs(delta_m)) < 1e-12:
            break
        elif np.max(np.abs(grad_v)) < 1e-12:
            break

    return history, cost_story, i

def Levenberg_Marquardt(stations:np.ndarray, event_test:np.ndarray, n_iteration:int,
                        vp:float=4000.0) -> tuple:
    """
    Function to prepare the gradient descent and compute what would block
    numba.

    Parameters
    ----------
    stations : numpy.ndarraynumpy.ndarray
        Position (X, Y and Z) and relative detection time.
        array([[X_i, Y_i, Z_i, t_i], ...])
    event_test : numpy.ndarray
        Initial event candidate.
        array([X, Y, Z, t])
    n_iteration : int
        Maximum number of iteration.
    vp : float, optional
        P-wave speed propagation in m/s. The default is 4000.0.

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
    history, cost_story, index = loop_Lev_Mar(stations, event_test,
                                              n_iteration, vp)

    history = history[:index+2]
    cost_story = cost_story[:index+2]
    best_idx = np.argmin(cost_story)
    misfit_fin = float(cost_story[best_idx])
    event_test = {'X':float(history[best_idx, 0]),
                  'Y':float(history[best_idx, 1]),
                  'Z':float(history[best_idx, 2]),
                  't':float(history[best_idx, 3])}

    return event_test, history, cost_story, misfit_fin
