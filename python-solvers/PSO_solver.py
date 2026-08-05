"""
Script for Particule Swarm Optimizer solver.

- calc_misfit_numba
    Compute RMSE misfit between expected and one observed arrival time.

- compute_loss_all
    Loop on all `samples` to compute their RMSE misfit between expected and
    observed arrival time.

- uniform_sampling
    Draw `n_samples` candidate events uniformly at random within the given
    per-dimension bounds.

- evolution_loop
    Run the evolution core loop of the optimizer with population generation
    until convergence.

- evolution
    Function to make a the search for the earthquake hypocenter parameters
    from a given data from stations, and and compute what would block numba.

"""

import math
import numpy as np
from numba import jit, njit, prange

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
def uniform_sampling(n_samples:int, limites:np.ndarray) -> np.ndarray:
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
    samples = np.random.uniform(0, 1, size=(n_samples, 4))
    samples = samples*(limites[:, 1]-limites[:, 0])+limites[:, 0]
    return samples

@jit(fastmath=True)
def evolution_loop(pop_size:int, limites:np.ndarray, stations:np.ndarray,
                   n_iteration:int, inertia:float=0.5, c_cogni:float=1.5,
                   c_social:float=1.5, patience:int=25, vp:float=4000.0
                   ) -> tuple:
    """
    Run the evolution core loop of the optimizer with population generation
    until convergence.

    Parameters
    ----------
    pop_size : int
        Length of the full population.
    limites : numpy.ndarray
        Parameters to parametrise the probability law we will sample.
    n_iteration : int
        Maximum number of iteration.
    inertia : float, optional
        Speed inertia factor, use to smooth the speed directions and
        amplitude between iterations, should be kept between 0.1 and 0.7. The
        default is 0.5.
    c_cogni : float, optional
        Cognitive coefficient, will weight the importance of the globale
        population direction of the search. The default is 1.5.
    c_social : float, optional
        Social coefficient, will weight the importance of the individual best
        sample of the population in the direction of the search. The default
        is 1.5.
    patience : int, optional
        Number of iterations to evaluate for early stopping criteria. The
        default is 25.
    vp : float, optional
        Assumed constant P-wave velocity. The default is 4000.0.

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
    p2p = int(2*patience)
    cost_story = np.empty(n_iteration+1)
    history = np.empty((n_iteration+1, 4))

    population = uniform_sampling(pop_size, limites)
    speeds = np.zeros((pop_size, 4))

    best_samples = np.copy(population)
    best_loss = compute_loss_all(population, stations)
    best_globale_pos = best_samples[np.argmin(best_loss)]
    best_globale_loss = np.min(best_loss)

    cost_story[0] = best_globale_loss
    history[0] = best_globale_pos
    for i in range(0, n_iteration, 1):
        r1 = np.random.rand(pop_size, 4)
        r2 = np.random.rand(pop_size, 4)

        speeds = (inertia*speeds +
                  c_cogni * r1 * (best_samples - population) +
                  c_social * r2 * (best_globale_pos - population))

        population += speeds
        loss = compute_loss_all(population, stations)

        mask = loss < best_loss
        best_samples[mask] = population[mask]
        best_loss[mask] = loss[mask]

        min_idx = np.argmin(best_loss)
        if best_loss[min_idx] < best_globale_loss:
            best_globale_pos = best_samples[min_idx]
            best_globale_loss = best_loss[min_idx]

        cost_story[i+1] = best_globale_loss
        history[i+1] = best_globale_pos
        if best_globale_loss < 1e-8:
            break

        if i > p2p:
            if (1-np.mean(cost_story[i-patience:i+1])/np.mean(
                          cost_story[i-p2p:i-patience])) <= 1e-6:
                break

    return history, cost_story, i

def evolution(stations:np.ndarray, limites:np.ndarray, n_iteration:int,
              pop_size:int, inertia:float=0.5, c_cogni:float=1.5,
              c_social:float=1.5, patience:int=10, vp:float=4000.0):
    """
    Function to make a the search for the earthquake hypocenter parameters
    from a given data from stations, and and compute what would block numba.

    Parameters
    ----------
    stations : numpy.ndarray
        2d array, normalised stations parameters, shape (N, 4).
    limites : numpy.ndarray
        Bounds of the parameter space to sample from, shape (4, 2): one
        `(lower, upper)` row per parameter (X, Y, Z, t).
    n_iteration : int
        Maximum number of iteration to run.
    pop_size : int
        Length of the full population.
    inertia : float, optional
        Speed inertia factor, use to smooth the speed directions and
        amplitude between iterations, should be kept between 0.1 and 0.7. The
        default is 0.5.
    c_cogni : float, optional
        Cognitive coefficient, will weight the importance of the globale
        population direction of the search. The default is 1.5.
    c_social : float, optional
        Social coefficient, will weight the importance of the individual best
        sample of the population in the direction of the search. The default
        is 1.5.
    patience : int, optional
        Number of iterations over which the early-stopping plateau
        criterion is evaluated (see `evolution_loop`). The default is 10.
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
    history, cost_story, index = evolution_loop(pop_size, limites, stations,
                                                n_iteration, inertia,
                                                c_cogni, c_social, patience,
                                                vp)

    history = history[:index+1]
    cost_story = cost_story[:index+1]
    best_idx = np.argmin(cost_story)
    misfit_fin = float(cost_story[best_idx])
    event_test = {'X':float(history[best_idx, 0]),
                  'Y':float(history[best_idx, 1]),
                  'Z':float(history[best_idx, 2]),
                  't':float(history[best_idx, 3])}

    return event_test, history, cost_story, misfit_fin
