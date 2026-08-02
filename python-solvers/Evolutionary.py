# -*- coding: utf-8 -*-
"""
Script for evolutionary solver.
 
- calc_misfit_numba:
    Compute RMSE misfit between expected and one observed arrival time.
 
- compute_loss_all:
    Loop on all `samples` to compute their RMSE misfit between expected and
    observed arrival time.
 
- randoms_childs:
    Draw uniformly-random candidate samples within given parameter bounds.
 
- selections:
    Select the best-scoring samples (survivors) from a population.
 
- cross_childs:
    Create offspring by randomly mixing coordinates between pairs of
    survivors.
 
- mutants_childs:
    Create offspring by perturbing survivors with correlated (covariance-
    based) Gaussian noise.
 
- evolution_loop:
    Run the generational loop of the genetic algorithm (selection,
    crossover, mutation, random re-injection, early stopping).
 
- evolution:
    Top-level entry point: build the crossover combination table, run
    `evolution_loop`, and return the best event candidate found.
 
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
        if samples_arr[i, 2] > 0:
            loss[i] = loss[i]*10

    return loss

@jit(fastmath=True)
def randoms_childs(shape:tuple, limits:np.ndarray) -> np.ndarray:
    """
    Draw new candidate samples uniformly at random within given per-
    parameter bounds. Used both to build the initial population and to
    re-inject fresh diversity into the population each generation.

    Parameters
    ----------
    shape : tuple
        Shape of the new samples randomly created: (number of samples, length
        of the samples).
    limits : numpy.ndarray
        Limits of the space to explore. The shape will be (n, 2) with n the
        number of parameters. For the n-th line, it is the (lower, higgher)
        bound. The default is None.

    Returns
    -------
    news : numpy.ndarray
        Listing of the new samples.

    """
    news = np.random.uniform(0.0, 1.0, size=shape)
    news = news * (limits[:, 1]-limits[:, 0]) + limits[:, 0]
    return news

@jit(fastmath=True)
def selections(score:np.ndarray, population:np.ndarray, nb_survivor:int
               ) -> np.ndarray:
    """
    Select the `nb_survivor` best-scoring (lowest-RMSE) samples from a
    population, sorted from best to worst.

    Uses `np.argpartition` to isolate the top `nb_survivor` samples in
    O(n) time, then sorts only that subset (O(k log k), k = nb_survivor)
    instead of sorting the full population — this gives the same ranking
    as a plain `np.argsort` over the whole population, but faster.

    Parameters
    ----------
    score : numpy.ndarray
        RMSE between station's data and the proposed samples.
    population : numpy.ndarray
        Candidate samples matching `score` order, shape (n, 4).
    nb_survivor : int
        Number of sample to keep (number of survivors).

    Returns
    -------
    survivors : numpy.ndarray
        The `nb_survivor` best samples, shape (nb_survivor, 4), sorted from
        best (lowest RMSE) to worst.
    rank : numpy.ndarray
        Indices into `population`/`score` of the selected samples, in the
        same best-to-worst order as `survivors`. `rank[0]` is the index of
        the single best sample in the input population.

    """
    idx = np.argpartition(score, nb_survivor)[:nb_survivor]
    rank = idx[np.argsort(score[idx])]
    survivors = population[rank]
    return survivors, rank

@jit(fastmath=True)
def cross_childs(survivors:np.ndarray, combinaisons:np.ndarray,
                 ) -> np.ndarray:
    """
    Create offspring via uniform crossover: randomly pair up survivors, and
    for each pair, build two children by swapping whole coordinates (X, Y,
    Z, t) between the two parents according to a randomly chosen mask.

    Because each coordinate is copied whole from one parent or the other
    (never blended), a child's coordinates always come from an already-valid
    parent value, which makes this operator safe to use on the correlated,
    non-linear depth/origin-time trade-off region of the misfit surface.

    Parameters
    ----------
    survivors : numpy.ndarray
        Samples kept from the `selections` function, shape (n, 4). `n` must
        be even.
    combinaisons : numpy.ndarray
        2d array of shape (6, 4), each row a binary mask over the 4
        coordinates indicating which parent (0 or 1) contributes that
        coordinate to the first child of the pair (the second child gets
        the complementary assignment).

    Returns
    -------
    news : numpy.ndarray
        The crossover offspring, same shape as `survivors` (n, 4): two
        children per parent pair.

    """
    length = len(survivors)
    half = length//2

    indices = np.arange(length)
    np.random.shuffle(indices)
    indices = indices.reshape(half, 2)
    indexes = combinaisons[np.random.randint(0, high=6, size=half)]
    news = np.empty((length, 4))
    for i in range(0, half, 1):
        m0 = indexes[i] == 0
        m1 = indexes[i] == 1
        news[i*2  , m0] = survivors[indices[i, 0], m0]
        news[i*2  , m1] = survivors[indices[i, 1], m1]
        news[i*2+1, m0] = survivors[indices[i, 1], m0]
        news[i*2+1, m1] = survivors[indices[i, 0], m1]

    return news

@jit(fastmath=True)
def mutants_childs(survivors:np.ndarray, n_childs:int):
    """
    Create mutant offspring from the survivor samples using correlated
    (covariance-based) by sampling a multi-variate normal distribution built
    from covariance (via Cholesky factor).. Each survivor produces exactly
    `n_childs` children.

    A variance floor of `1e-16` is enforced on the diagonal of the
    covariance matrix, to keep the Cholesky decomposition well-defined if a
    dimension's variance collapses to (near) zero across the survivor
    population.

    Parameters
    ----------
    survivors : numpy.ndarray
        Samples kept from the `selections` function, shape (n, 4).
    num_childs : int
        Number of mutant childs per survivor.

    Returns
    -------
    news : numpy.ndarray
        The mutated samples, shape (n_childs * n, 4).

    """
    n, d = survivors.shape
    mean = np.empty(d)
    for j in range(d):
        mean[j] = np.mean(survivors[:, j])

    centered = survivors - mean
    cov = (centered.T @ centered) / (n - 1)
    for j in range(d):
        if cov[j, j] < 1e-16:
            cov[j, j] = 1e-16

    L = np.linalg.cholesky(cov)
    z = np.random.randn(n_childs*n, d)
    news = survivors[np.arange(0, n_childs*n, 1)%n] + z @ L.T
    return news

@jit(fastmath=True)
def evolution_loop(pop_size:int, limites:np.ndarray, stations:np.ndarray,
                   n_iteration:int, p_surv:float, combinaisons:np.ndarray,
                   patience:int=10, vp:float=4000.0) -> tuple:
    """
    Run the generational loop of the genetic algorithm: score, select,
    recombine (crossover + covariance mutation), re-inject random samples,
    and repeat, until `n_iteration` is reached or an early-stopping
    criterion is met.

    Each generation's population of size `pop_size` is rebuilt as:
    `num_surv` survivors + `num_surv` crossover children + `3*num_surv`
    mutant children (3 mutants per survivor) + `re_rand` fresh random
    samples, where `num_surv = p_surv * pop_size` (rounded down to an even
    number) and `re_rand = pop_size - 5*num_surv`. For `re_rand` to stay
    positive (so the population is refreshed with new random samples every
    generation), `p_surv` must stay below `1/5 = 0.2`; at or above that,
    `re_rand` is skipped and the generation's population shrinks below
    `pop_size`.

    Two independent early-stopping checks can end the loop before
    `n_iteration` is reached:
    - Absolute: if the best sample's RMSE drops to `1e-8` or below.
    - Plateau: every `patience` iterations (once past `2*patience`
      iterations), if the mean best-RMSE over the last `patience`
      iterations has improved by less than `1e-6` (relative) compared to
      the `patience` iterations before that.

    Parameters
    ----------
    pop_size : int
        Length of the full population.
    limites : numpy.ndarray
        Bounds of the parameter space to sample from, shape (4, 2): one
        `(lower, upper)` row per parameter (X, Y, Z, t).
    stations : numpy.ndarray
        2d array, normalised stations parameters, shape (N, 4).
    n_iteration : int
        Maximum number of generations to run.
    p_surv : float
        Fraction of `pop_size` kept as survivors each generation. Must be
        below `1/5 = 0.2` for the population to be refreshed with random
        samples every generation (see above).
    combinaisons : numpy.ndarray
        2d array of shape (6, 4) of crossover masks, passed through to
        `cross_childs`.
    patience : int, optional
        Number of iterations over which the plateau early-stopping
        criterion is evaluated. The default is 10.
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
    num_surv = int(p_surv*pop_size)
    num_surv = num_surv - num_surv%2
    re_rand = int(pop_size-num_surv*5)

    cost_story = np.empty(n_iteration)
    history = np.empty((n_iteration, 4))

    population = randoms_childs((pop_size, 4), limites)
    for i in range(0, n_iteration, 1):
        scores = compute_loss_all(population, stations, vp)
        survivants, rank = selections(scores, population, num_surv)
        cost_story[i] = scores[rank[0]]
        history[i] = survivants[0]
        if cost_story[i] <= 1e-8:
            break

        if i > p2p:
            # Early stoping
            if (1-np.mean(cost_story[i-patience:i+1])/np.mean(
                          cost_story[i-p2p:i-patience])) <= 1e-6:

                break

        crossover = cross_childs(survivants, combinaisons)
        mutors = mutants_childs(survivants, 3)
        if re_rand > 0:
            rands = randoms_childs((re_rand, 4), limites)
            population = np.concatenate((survivants, crossover, mutors,
                                         rands))
        else:
            population = np.concatenate((survivants, crossover, mutors))

    return history, cost_story, i

def evolution(pop_size, limites, n_iteration, stations, p_surv,
              patience:int=10, vp:float=4000.0):
    """
    Search for the earthquake hypocenter parameters (X, Y, Z, t) that best
    fit observed station arrival times, via a genetic algorithm.

    Builds the fixed crossover combination table, delegates the
    generational search to `evolution_loop`, then extracts and returns the
    best candidate found over the course of the run (which is not always
    the candidate from the very last completed generation, so the true
    minimum is taken over the full `cost_story`).

    Parameters
    ----------
    pop_size : int
        Length of the full population.
    limites : numpy.ndarray
        Bounds of the parameter space to sample from, shape (4, 2): one
        `(lower, upper)` row per parameter (X, Y, Z, t).
    n_iteration : int
        Maximum number of generations to run.
    stations : numpy.ndarray
        2d array, normalised stations parameters, shape (N, 4).
    p_surv : float
        Fraction of `pop_size` kept as survivors each generation. Each
        generation then consists of: `num_surv` survivors + `num_surv`
        crossover children + `3*num_surv` mutant children + the remaining
        samples drawn at random, where `num_surv = p_surv * pop_size`
        (rounded down to an even number). For the random-sample slot to
        stay positive, `p_surv` must be below `1/5 = 0.2`.
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
    combinaisons = np.array([[0, 0, 1, 1], [0, 1, 0, 1], [1, 0, 0, 1],
                             [0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 0]])

    history, cost_story, index = evolution_loop(
        pop_size, limites, stations, n_iteration, p_surv,
        combinaisons, patience, vp)

    history = history[:index+1]
    cost_story = cost_story[:index+1]
    best_idx = np.argmin(cost_story)
    misfit_fin = float(cost_story[best_idx])
    event_test = {'X':float(history[best_idx, 0]),
                  'Y':float(history[best_idx, 1]),
                  'Z':float(history[best_idx, 2]),
                  't':float(history[best_idx, 3])}

    return event_test, history, cost_story, misfit_fin
