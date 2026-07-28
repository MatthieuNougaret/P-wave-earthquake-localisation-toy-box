
"""
Script for Gauss-Newton solver.
- compute_residuals:
    Compute the residuals and distances between the candidate and the
    stations.

- compute_Jacobian:
    Compute the Jacobian matrix.

- loop_Gauss_Newton:
    Search the hypocenter with gradient descent

- Gauss_Newton:
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
def loop_Gauss_Newton(stations:np.ndarray, event_test:np.ndarray,
                      n_iteration:int, steps:np.ndarray, vp:float=4000.0
                      ) -> tuple:
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
    steps : numpy.ndarray
        Vector with the size of the step per dimensions (0:X, 1:Y, 2:Z, 3:t).
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
    dampling = np.eye(4, 4)
    dampling[0, 0] = steps[0]
    dampling[1, 1] = steps[1]
    dampling[2, 2] = steps[2]
    dampling[3, 3] = steps[3]

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
        Hess_LM = Hess + dampling
        delta_m = np.linalg.solve(Hess_LM, -grad_v)

        tested = event_test + delta_m
        if tested[2] > 0:
            depth_res *= 1.5
            event_test[2] = start_point[2] * depth_res
            if event_test[2] > -1:
                event_test[2] = -1

            res, dist = compute_residuals(stations, event_test, vp)
            misfit_current = np.sum(res**2)/2

            dampling[2, 2] = steps[2]

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
                dampling = dampling / factor

            else:
                dampling = dampling * factor

        history[i+1] = event_test
        cost_story[i+1] = math.sqrt(np.sum(res**2 /n_stations))
        if np.max(np.abs(delta_m)) < 1e-12:
            break
        elif np.max(np.abs(grad_v)) < 1e-12:
            break

    return history, cost_story, i

def Gauss_Newton(stations:np.ndarray, event_test:np.ndarray, n_iteration:int,
                 steps:np.ndarray=np.array([1, 1, 1, 0.001]), vp:float=4000.0
                 ) -> tuple:
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
    steps : numpy.ndarray, optional
        Vector with the size of the step per dimensions (0:X, 1:Y, 2:Z, 3:t).
        The default is np.array([1, 1, 1, 0.001]).
    vp : float, optional
        P-wave speed propagation in m/s. The default is 4000.0.

    Returns
    -------
    event_test : dictionary
        Best event candidate (lowest RMSE).
        {X:float, Y:float, Z:float, t:float}
    history : numpy.ndarray
        2d array with the history of the best event candidate.
    cost_story : numpy.ndarray
        Vector with the history of the RMSE of the event candidate.
    misfit_fin : float
        RMSE of `event_test`, the lowest misfit found during the search.

    """
    history, cost_story, index = loop_Gauss_Newton(stations, event_test,
                                                   n_iteration, steps, vp)

    history = history[:index+2]
    cost_story = cost_story[:index+2]
    best_idx = np.argmin(cost_story)
    misfit_fin = float(cost_story[best_idx])
    event_test = {'X':float(history[best_idx, 0]),
                  'Y':float(history[best_idx, 1]),
                  'Z':float(history[best_idx, 2]),
                  't':float(history[best_idx, 3])}

    return event_test, history, cost_story, misfit_fin
