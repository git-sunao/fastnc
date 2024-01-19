import numpy as np

def loglinear(logmin, logmax, linmin, linmax, nbinlog, nbinlin):
    """
    Create a log-linear binning.

    Parameters
    ----------
    logmin : float
        The minimum value of the log binning.
    logmax : float
        The maximum value of the log binning.
    linmin : float
        The minimum value of the linear binning.
    linmax : float
        The maximum value of the linear binning.
    nbinlog : int
        The number of bins in the log binning.
    nbinlin : int
        The number of bins in the linear binning.

    Returns
    -------
    out : ndarray
        An array of shape (nbinlog + nbinlin,) containing the bin edges.
    """
    logmin = np.log10(logmin)
    logmax = np.log10(logmax)
    out = np.hstack([np.logspace(logmin, logmax, nbinlog), np.linspace(linmin, linmax, nbinlin)])
    return out

def edge_correction(x, vmin, vmax, atol=1e-8, rtol=1e-5):
    """
    Replace values in x that are out of bounds 
    but close to the boundary edges.

    This is used to correct the x values that get out of bounds
    due to numerical errors.

    Parameters
    ----------
    x : ndarray
        Input array.
    vmin : float
        Minimum value.
    vmax : float
        Maximum value.

    Returns
    -------
    out : ndarray
        Output array.
    """
    out = x.copy()
    out[np.logical_and(np.isclose(x, vmin, atol, rtol), x<vmin)] = vmin
    out[np.logical_and(np.isclose(x, vmax, atol, rtol), x>vmax)] = vmax
    return out