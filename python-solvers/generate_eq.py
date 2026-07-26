
"""
Script to generate random earthquakes position and timing.

"""

import numpy as np
from time import time
from copy import deepcopy

def generate_event(stations:dict|np.ndarray, limites:np.ndarray,
                   vp:float=4000.0, noise_prop:float=0.0) -> tuple:
    """
    Function to generate a random earthquake position and fill the stations
    detection time and return it.

    Parameters
    ----------
    stations : dict | numpy.ndarray
        Position of the stations. Can be a dictionary or numpy.ndarray.
        If dico:
            {station_i:{X:float, Y:float, Z:float, t:0.0}, ...}
        If ndarray:
            np.array([[float, float, float, 0.0], ...])
    limites : numpy.ndarray
        World box limites in which the earthquake can happen. A 2 columns
        array with minimum and maximum bounds. ([[min_X, max_X], [min_Y,
        max_Y], [min_Z, max_Z]]).
    vp : float, optional
        P-wave propagation speed. The default is 4000.0.
    noise_prop : float, optional
        Parameter to induce noise in the detection time. It will be (if > 0)
        proportional to the hypocenter-station distance. The default is 0.0.

    Returns
    -------
    event : dictionary
        X, Y and Z hypocenter position and t the time of the failure.
    event_arr : numpy.ndarray
        Hypocenter position and time of the failure with indexes 0:X, 1:Y,
        2:Z and 3:t.
    stations : dictionary or numpy.ndarray
        Will be of the same type as the given `stations`. Stations with the
        noisy detection time. If noise_prop = 0.0, will be equal to
        stations_true.
    stations_true : dictionary or numpy.ndarray
        Will be of the same type as the given `stations`. Stations with the
        detection time without any noise added.

    """
    delta_box = limites[:3, 1] - limites[:3, 0]
    position = limites[:3, 0] + np.random.rand(3) * delta_box

    event = {'X': float(position[0]),
             'Y': float(position[1]),
             'Z': float(position[2]),
             't': float(time())}

    event_arr = np.array(list(event.values()))
    stations_true = deepcopy(stations)
    if type(stations) == dict:
        for key in stations:
            dist = ((event['X']-stations[key]['X'])**2
                   +(event['Y']-stations[key]['Y'])**2
                   +(event['Z']-stations[key]['Z'])**2)**0.5

            noise = np.random.randn() * dist / vp * noise_prop

            stations_true[key]['t'] = event['t'] + dist / vp
            stations[key]['t'] = event['t'] + dist / vp + noise

    elif type(stations) == np.ndarray:
        dists = np.sum((event_arr[:3]-stations[:, :3])**2, axis=1)**0.5
        noises = np.random.randn(len(stations)) * dists / vp * noise_prop
        stations_true[:, 3] = event['t'] + dists / vp
        stations[:, 3] = event['t'] + dists / vp + noise

    else:
        raise TypeError(f'`stations` must be either `dict` or `np.ndarray`' \
                        f', found {type(stations)}')

    return event, event_arr, stations, stations_true
