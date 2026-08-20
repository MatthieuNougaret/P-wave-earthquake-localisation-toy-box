
"""
Script for signal treatment used before the call of a solver.

- centrering:
    To normalise (centralise) stations and event if given from a specific
    station.

- centrering_arr:
    To normalise (centralise) stations and event (if given) using the station
    which is the closet to the geometric center of the network.

"""

import math
import numpy as np
from numba import jit
from copy import deepcopy

def centrering(stations:dict|np.ndarray, key:str|int,
               event:dict|np.ndarray=None) -> tuple|dict|np.ndarray:
    """
    Function to normalise (centralise) stations and event if given.

    Parameters
    ----------
    stations : dictionary or numpy.ndarray
        Station locations and picked arrival time.
    key : str or int
        Item key to select the reference station.
    event : dictionary or numpy.ndarray
        Event to be normalised with the key reference station. The default
        is None.

    Returns
    -------
    stations : dictionary or numpy.ndarray
        Normalised (centralised) station relative to the reference station.
    event : dictionary or numpy.ndarray, optional
        Normalised (centralised) event relative to the reference station.

    """
    if type(event) == dict:
        site_ref = deepcopy(stations[key])
        event['X'] = event['X']-site_ref['X']
        event['Y'] = event['Y']-site_ref['Y']
        event['Z'] = event['Z']-site_ref['Z']
        event['t'] = event['t']-site_ref['t']

    elif type(event) == np.ndarray:
        event = event-stations[key]

    if type(stations) == dict:
        site_ref = deepcopy(stations[key])
        for i in stations:
            stations[i]['X'] = stations[i]['X']-site_ref['X']
            stations[i]['Y'] = stations[i]['Y']-site_ref['Y']
            stations[i]['t'] = stations[i]['t']-site_ref['t']

        if type(event) == dict:
            return stations, event
        else:
            return stations

    elif type(stations) == np.ndarray:
        stations = stations-stations[key]
        if type(event) == np.ndarray:
            return stations, event
        else:
            return stations

    else:
        raise TypeError('stations must be dict or numpy.ndarray.')

def centrering_arr(stations:dict|np.ndarray, event:dict|np.ndarray=None
                   ) -> tuple|np.ndarray:
    """
    Function to normalise stations and event (if given) using the station
    which is the closet to the geometric center of the network.

    Parameters
    ----------
    stations : dictionary or numpy.ndarray
        Station locations and picked arrival time.
    event : NoneType or dictionary or np.ndarray
        Event to be normalised. The default is None.

    Returns
    -------
    stations : numpy.ndarray
        Normalised station relative to the reference station.
    event : numpy.ndarray, optional
        Normalised event relative to the reference station.

    """
    if type(stations) == np.ndarray:
        stations_arr = np.copy(stations)
    elif type(stations) == dict:
        stations_arr = np.array([
            list(sub.values()) for sub in stations.values()])

    mlt_d = ((stations_arr[:, 0]-stations_arr[:, None, 0])**2 + (
              stations_arr[:, 1]-stations_arr[:, None, 1])**2)**0.5

    index = np.argmin(np.sum(mlt_d, axis=0))

    # most centered station
    if event is not None:
        if type(event) == np.ndarray:
            event_arr = np.copy(event)
        elif type(event) == dict:
            event_arr = np.array(list(event.values()))

        event_arr = event_arr-stations_arr[index]

    stations_arr = stations_arr-stations_arr[index]
    if event is not None:
        return stations_arr, event_arr
    else:
        return stations_arr


def distance_dict2dict(dico_1:dict, dico_2:dict) -> float:
    """
    Function to compute the distance between to position stored into
    dictionaries.

    Parameters
    ----------
    dico_1 : dictinary
        First position.
    dico_2 : dictinary
        Second position.

    Returns
    -------
    d : float
        Distance between the position stored into the dictionaries.

    """
    d = math.sqrt((dico_2['X'] - dico_1['X'])**2 +
                  (dico_2['Y'] - dico_1['Y'])**2 +
                  (dico_2['Z'] - dico_1['Z'])**2)

    return d
