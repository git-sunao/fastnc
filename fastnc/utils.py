import numpy as np
import pickle

def loglinear(logmin, logmax, linmin, linmax, nbinlog, nbinlin):
    """
    Create a log-linear binning.

    Parameters:
    logmin (float): The minimum value for the logarithmic range.
    logmax (float): The maximum value for the logarithmic range.
    linmin (float): The minimum value for the linear range.
    linmax (float): The maximum value for the linear range.
    nbinlog (int): The number of bins in the logarithmic range.
    nbinlin (int): The number of bins in the linear range.

    Returns:
    numpy.ndarray: An array containing the log-linear binning.
    """
    logmin = np.log10(logmin)
    logmax = np.log10(logmax)
    out = np.hstack([np.logspace(logmin, logmax, nbinlog), np.linspace(linmin, linmax, nbinlin)])
    return out

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