import numpy as np

def align_and_rmse(x1, y1, x2, y2, n_points=100):
    """
    Interpolate two curves onto a common x-axis, then compute RMSLE.

    Parameters
    ----------
    x1, y1 : array-like
        X and Y values for the first curve.
    x2, y2 : array-like
        X and Y values for the second curve.
    n_points : int, optional
        Number of points in the common axis. If None, uses the union of x1 and x2.

    Returns
    -------
    common_x : np.ndarray
        Shared x-axis.
    y1_interp : np.ndarray
        First curve interpolated onto common_x.
    y2_interp : np.ndarray
        Second curve interpolated onto common_x.
    rmsle : float
        Root mean squared log error between the interpolated arrays.
    """
    
    x1 = np.asarray(x1, dtype=float)
    y1 = np.asarray(y1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    y2 = np.asarray(y2, dtype=float)

    if x1.ndim != 1 or y1.ndim != 1 or x2.ndim != 1 or y2.ndim != 1:
        raise ValueError("All inputs must be 1D arrays.")
    if len(x1) != len(y1) or len(x2) != len(y2):
        raise ValueError("Each x array must have the same length as its y array.")

    # Sort by x so interpolation works correctly
    idx1 = np.argsort(x1)
    idx2 = np.argsort(x2)

    x1, y1 = x1[idx1], y1[idx1]
    x2, y2 = x2[idx2], y2[idx2]

    # Keep only overlapping region
    xmin = max(x1.min(), x2.min())
    xmax = min(x1.max(), x2.max())

    if xmin >= xmax:
        raise ValueError("The two x-axes do not overlap.")

    common_x = np.logspace(np.log10(xmin),
                           np.log10(xmax),
                           (np.log10(xmax)-np.log10(xmin))*100+1)

    y1_interp = np.interp(common_x, x1, y1)
    y2_interp = np.interp(common_x, x2, y2)

    rmse = np.sqrt(np.mean(y1_interp - y2_interp) ** 2)

    return common_x, y1_interp, y2_interp, rmse