import numpy as np
import logging
import math

logger = logging.getLogger(__name__)
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)

def get_velocity(angle, time):
    # angle is assumed unwrapped
    angle = np.array(angle)
    time = np.array(time)
    angular_velocity = np.divide(np.gradient(angle), np.gradient(time))

    return np.array(angular_velocity)

# Source: https://scipy.github.io/old-wiki/pages/Cookbook/SavitzkyGaolay
# Edited
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """

    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    #except ValueError, msg:
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * math.factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def savitzky_golay_np2(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """

    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    #except ValueError, msg:
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    # Use np.asmatrix instead of np.mat for numpy 2.0
    b = np.asmatrix([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * math.factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def align_timestamps_nw(x, y, new, match = 1, mismatch = 1, gap = 1, thresh=0.1):
    """
    Align sets of digital timestamps using the Needleman-Wunsch algorithm.

    Compares between timestamp difference vectors. If the differences go over
    threshold, it is not a match. Constructs a scoring matrix and a direction
    matrix for the NW algorithm, and then computes a maximum score path
    through the scoring matrix. The path is decoded to form the optimal
    alignment. Good differences are used to form match vectors and a polynomial is
    fitted to determine drift of timestamps relative to each other.

    References: 
    http://en.wikipedia.org/wiki/Needleman%E2%80%93Wunsch_algorithm
    http://www.avatar.se/molbioinfo2001/dynprog/dynamic.html
    http://www.hrbc-genomics.net/training/bcd/Curric/PrwAli/node3.html

    Adapted from Kamil Slowikowski (github: slowkow) Python implementation of NW
    https://gist.github.com/slowkow/06c6dba9180d013dfd82bec217d22eb5

    using improved algorithm from previously developed MATLAB version (Manu Madhav, 2015)
    https://www.mathworks.com/matlabcentral/fileexchange/52819-align_timestamps

    Manu Madhav
    25-Apr-2023
    """

    logger.info('Aligning timestamps of length %d and %d', len(x), len(y))

    dx = np.diff(x)
    dy = np.diff(y)

    nx = len(dx)
    ny = len(dy)

    # Optimal score at each possible pair of characters.
    F = np.zeros((nx + 1, ny + 1))
    F[:,0] = np.linspace(0, -nx * gap, nx + 1)
    F[0,:] = np.linspace(0, -ny * gap, ny + 1)

    # Pointers to trace through an optimal aligment.
    P = np.zeros((nx + 1, ny + 1))
    P[:,0] = 3
    P[0,:] = 4

    # Temporary scores.
    t = np.zeros(3)
    for i in range(nx):
        for j in range(ny):
            if abs(dx[i] - dy[j])<=thresh:
                t[0] = F[i,j] + match
            else:
                t[0] = F[i,j] - mismatch
            t[1] = F[i,j+1] - gap
            t[2] = F[i+1,j] - gap
            tmax = np.max(t)
            F[i+1,j+1] = tmax
            if t[0] == tmax:
                P[i+1,j+1] += 2
            if t[1] == tmax:
                P[i+1,j+1] += 3
            if t[2] == tmax:
                P[i+1,j+1] += 4

    # Trace through an optimal alignment.
    i = nx
    j = ny
    # Indices of optimal alignment
    x_idx = np.array([])
    y_idx = np.array([])
    while i > 0 or j > 0:
        if P[i,j] in [2, 5, 6, 9]:
            x_idx = np.append(x_idx, i-2)
            y_idx = np.append(y_idx, j-2)
            i -= 1
            j -= 1
        elif P[i,j] in [3, 5, 7, 9]:
            x_idx = np.append(x_idx, i-2)
            y_idx = np.append(y_idx, np.nan)
            i -= 1
        elif P[i,j] in [4, 6, 7, 9]:
            x_idx = np.append(x_idx, np.nan)
            y_idx = np.append(y_idx, j-2)
            j -= 1

    # Reverse the strings.
    x_idx = np.flip(x_idx)
    y_idx = np.flip(y_idx)

    good_diffs_idx = np.logical_not(np.logical_or(np.isnan(x_idx), np.isnan(y_idx)))

    x_match_idx = np.union1d(x_idx[good_diffs_idx], np.add(x_idx[good_diffs_idx],1)).astype(int)[1:]
    y_match_idx = np.union1d(y_idx[good_diffs_idx], np.add(y_idx[good_diffs_idx],1)).astype(int)[1:]

    x_match = x[x_match_idx]
    y_match = y[y_match_idx]

    logger.info('Found %d matching timestamps', len(x_match))

    if not new:
    # Polynomial so that y_ts = np.polyval(p, x_ts) + x_ts
        p = np.polyfit(x_match, y_match-x_match, 1)

    else:
    # Polynomial so that y_ts = np.polyval(p, x_ts)
        p = np.polyfit(x_match, y_match, 1)

    # two different polyfit calculation yields similar y_ts ??
    # For example, in rat 10 session 230426
    # math.isclose(new_spike_timestamps_in_ros_frame[i], spike_timestamps_in_ros_frame[i], rel_tol=1e-9)
    # is true for all timestamps, while relative tolerance is smaller for interpolated sound degree
    # math.isclose(new_spike_soundDegree_sync[i], spike_soundDegree_sync[i], rel_tol=1e-7)
    # is true for all timestamps. If set rel_tol=1e-8, 2 timestamps have it to be false
    
    return p, x_match, y_match
