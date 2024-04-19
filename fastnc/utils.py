#!/usr/bin/env python
'''
Author     : Sunao Sugiyama 
Last edit  : 2024/03/25 21:00:58

Description:
utils.py contains utility functions for fastnc.
'''
import numpy as np
import fcntl
import os

# file locking
def acquire_lock(lock_file):
    lock_fd = open(lock_file, 'w')
    fcntl.flock(lock_fd, fcntl.LOCK_EX)

def release_lock(lock_file):
    lock_fd = open(lock_file, 'w')
    fcntl.flock(lock_fd, fcntl.LOCK_UN)

def npload_lock(filename, suffix='.npz'):
    lock_file = filename.replace(suffix, '.lock')
    acquire_lock(lock_file)
    try:
        data = np.load(filename)
    finally:
        release_lock(lock_file)
    return data

def npsavez_lock(filename, data, suffix='.npz'):
    lock_file = filename.replace(suffix, '.lock')
    acquire_lock(lock_file)
    try:
        np.savez(filename, **data)
    finally:
        release_lock(lock_file)

# Binning utilities
def loglinear(xmin, xmid, xmax, nbin1, nbin2):
    """
    Create a log-linear binning.

    Parameters:
        xmin (float) : The minimum value.
        xmid (float) : The value where the logarithmic and linear ranges meet.
        xmax (float) : The maximum value.
        nbin1 (int)  : The number of bins in the logarithmic range.
        nbin2 (int)  : The number of bins in the linear range.

    Returns:
        numpy.ndarray: An array containing the log-linear binning.
    """
    xbin1 = np.logspace(np.log10(xmin), np.log10(xmid), nbin1+1)[:-1]
    xbin2 = np.linspace(xmid, xmax, nbin2)
    xbin  = np.hstack([xbin1, xbin2])
    return xbin
    
def edge_correction(x, vmin, vmax, atol=1e-8, rtol=1e-5):
    """
    Replace values in x that are out of bounds but close to the boundary edges.
    This is useful for avoiding numerical errors triangle parameter conversion.
    Triangle is parametrized in multiple ways, 

    Parameters:
        x (ndarray)  : Input array.
        vmin (float) : Minimum value.
        vmax (float) : Maximum value.

    Returns:
        out (ndarray): Output array.
    """
    out = x.copy()
    out[np.logical_and(np.isclose(x, vmin, atol, rtol), x<vmin)] = vmin
    out[np.logical_and(np.isclose(x, vmax, atol, rtol), x>vmax)] = vmax
    return out

# Triangle phase factor
def sincos2angbar(psi, delta):
    """
    Calculate triangle phase factor 
    in sine and cosine form.

    Parameters:
        psi (float)   : psi.
        delta (float) : delta.
    """
    cos2b = np.cos(delta) + np.sin(2*psi)
    sin2b = np.cos(2*psi) * np.sin(delta)
    norm  = np.sqrt(cos2b**2 + sin2b**2)
    return sin2b/norm, cos2b/norm

# Config utilities
def merge_config_kwargs(config=None, **kwargs):
    """
    Merge the configuration dictionary and the keyword arguments.

    Parameters:
        config (dict) : Configuration dictionary.
        **kwargs      : Keyword arguments.
    """
    return {**(config or {}), **kwargs}

def get_config_key(config, key, default=None, **kwargs):
    """
    Get the value of a key from the configuration dictionary.

    Parameters:
        config (dict) : Configuration dictionary.
        key (str)     : Key.
        default       : Default value.
        **kwargs      : Keyword arguments.
    """
    config = merge_config_kwargs(config, **kwargs)
    return config.get(key, default)

def update_config(config_base, config=None, **kwargs):
    """
    Update the configuration dictionary.

    Parameters:
        config_base (dict) : Base configuration dictionary.
        config (dict)      : Configuration dictionary.
        **kwargs           : Keyword arguments.
    """
    config = merge_config_kwargs(config, **kwargs)
    config_base.update((key, value) for key, value in config.items() \
            if key in config_base)
