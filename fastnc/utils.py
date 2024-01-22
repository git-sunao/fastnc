#!/usr/bin/env python
'''
Description:
utils.py contains utility functions for fastnc.

Author     : Sunao Sugiyama 
Last edit  : 2024/01/21 21:12:13
'''
import numpy as np
import pickle

def loglinear(xmin, xmid, xmax, nbin1, nbin2):
    """
    Create a log-linear binning.

    Parameters:
    xmin (float): The minimum value.
    xmid (float): The value where the logarithmic and linear ranges meet.
    xmax (float): The maximum value.
    nbin1 (int): The number of bins in the logarithmic range.
    nbin2 (int): The number of bins in the linear range.

    Returns:
    numpy.ndarray: An array containing the log-linear binning.
    """
    xbin1 = np.logspace(np.log10(xmin), np.log10(xmid), nbin1+1)[:-1]
    xbin2 = np.linspace(xmid, xmax, nbin2)
    xbin  = np.hstack([xbin1, xbin2])
    return xbin
    
def edge_correction(x, vmin, vmax, atol=1e-8, rtol=1e-5):
    """Replace values in x that are out of bounds but close to the boundary edges.

    Parameters:
    x (ndarray): Input array.
    vmin (float): Minimum value.
    vmax (float): Maximum value.

    Returns:
    out (ndarray): Output array.
    """
    out = x.copy()
    out[np.logical_and(np.isclose(x, vmin, atol, rtol), x<vmin)] = vmin
    out[np.logical_and(np.isclose(x, vmax, atol, rtol), x>vmax)] = vmax
    return out

def save_pickle(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj