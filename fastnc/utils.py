import numpy as np

def err_fnc_diff_by_norm(out1, out2):
    """
    Error function for adaptive integration.
    """
    diff = np.mean(np.abs(out1-out2))
    mean = np.max(np.abs(out1))
    return diff/mean

def aint(fnc, xmin, xmax, Nx=2, axis=0, tol=1e-3, max_itern=10, err_fnc=None, verbose=False, **fnc_args):
    """
    Adaptive integration routine.

    Parameters
    ----------
    fnc : function
        Function to integrate.
    xmin : float
        Lower limit of integration.
    xmax : float
        Upper limit of integration.
    Nx : int
        Initial number of bins.
    axis : int
        Axis along which to integrate.
    tol : float
        Tolerance for error.
    max_itern : int
        Maximum number of iterations.
    err_fnc : function
        Function to calculate the error.
        Default is err_fnc_diff_by_norm.
    verbose : bool

    Returns
    -------
    out : float
        Integral.
    """

    if err_fnc is None:
        err_fnc = err_fnc_diff_by_norm

    # helper function updating the integral
    def helper(x, out):
        dx = x[1]-x[0]
        xval = x + dx/2
        fval = fnc(xval, **fnc_args)
        out  = (out + np.sum(fval, axis=axis)*dx)/2.0
        x = np.sort(np.hstack([x, xval]))
        return x, out

    # initialize
    x = np.linspace(xmin, xmax, Nx+1)[:-1]
    out = np.sum(fnc(x, **fnc_args), axis=axis) * (x[1]-x[0])
    err = 1e3
    itern = 0

    # iterate until convergence
    while (err > tol) and (itern <= max_itern):
        x, out2 = helper(x, out)
        err = err_fnc(out, out2)
        out = out2
        itern += 1
        
    # warn if max iterations reached
    converged = err <= tol
    if (not converged) and verbose:
        print('Warning: max iterations reached. err={err:.2e}, tol={tol:.2e}'.format(err=err, tol=tol))

    return out, converged

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