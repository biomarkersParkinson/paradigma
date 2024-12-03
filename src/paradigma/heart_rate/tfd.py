import numpy as np
from scipy import signal

"""

This module contains the implementation of the Generalized Time-Frequency Distribution (TFD) computation using non-separable kernels.
This is a Python implementation of the MATLAB code provided by John O Toole in the following repository: https://github.com/otoolej/memeff_TFDs 

The following functions are implemented for the computation of the TFD:
    - nonsep_gdtfd: Computes the generalized time-frequency distribution using a non-separable kernel.
    - get_analytic_signal: Generates the analytic signal of the input signal.
    - gen_analytic: Generates the analytic signal by zero-padding and performing FFT.
    - gen_time_lag: Generates the time-lag distribution of the analytic signal.
    - multiply_kernel_signal: Multiplies the TFD by the Doppler-lag kernel.
    - gen_doppler_lag_kern: Generates the Doppler-lag kernel based on kernel type and parameters.
    - get_kern: Gets the kernel based on the provided kernel type.
    - get_window: General function to calculate a window function.
    - get_win: Helper function to create the specified window type.
    - shift_window: Shifts the window so that positive indices appear first.
    - pad_window: Zero-pads the window to a specified length.
    - compute_tfd: Finalizes the time-frequency distribution computation.


"""	

def nonsep_gdtfd(x, kern_type=None, kern_params=None):
    """
    Computes the generalized time-frequency distribution (TFD) using a non-separable kernel.

    Parameters:
    -----------
    x : ndarray
        Input signal to be analyzed.
    kern_type : str, optional
        Type of kernel to be used for TFD computation. Default is None.
    kern_params : list, optional
        List of parameters specific to the kernel type. Default is None.

    Returns:
    --------
    tfd : ndarray
        The computed time-frequency distribution.
    """
    z = get_analytic_signal(x)
    N = len(z) // 2  # Since z is a signal of length 2N
    Nh = int(np.ceil(N / 2))

    # Generate the time-lag distribution of the analytic signal
    tfd = gen_time_lag(z)

    # Multiply the TFD by the Doppler-lag kernel
    tfd = multiply_kernel_signal(tfd, kern_type, kern_params, N, Nh)
    
    # Finalize the TFD computation
    tfd = compute_tfd(N, Nh, tfd)
    
    return tfd

def get_analytic_signal(x):
    """
    Generates the signals analytic version.

    Parameters:
    -----------
    z : ndarray
        Input real-valued signal.

    Returns:
    --------
    z : ndarray
        Analytic signal with zero-padded imaginary part.
    """
    N = len(x)

    # Ensure the signal length is even by trimming one sample if odd, since the gen_time_lag function requires an even-length signal
    if N % 2 != 0:
        x = x[:-1]

    # Make the analytical signal of the real-valued signal z (preprocessed PPG signal)
    # doesn't work for input of complex numbers
    z = gen_analytic(x)  
    
    return z

def gen_analytic(x):
    """
    Generates an analytic signal by zero-padding and performing FFT.

    Parameters:
    -----------
    x : ndarray
        Input real-valued signal.

    Returns:
    --------
    z : ndarray
        Analytic signal in the time domain with zeroed second half.
    """
    N = len(x)
    
    # Zero-pad the signal to double its length
    x = np.concatenate((np.real(x), np.zeros(N)))
    x_fft = np.fft.fft(x)

    # Generate the analytic signal in the frequency domain
    H = [1] + list(np.repeat(2, N-1)) + [1] + list(np.repeat(0, N-1))
    z_cb = np.fft.ifft(x_fft * H)

    # Force the second half of the time-domain signal to zero
    z = np.concatenate((z_cb[:N], np.zeros(N)))

    return z

def gen_time_lag(z):
    """
    Generate the time-lag distribution of the analytic signal z.

    Parameters:
    -----------
    z : ndarray
        Analytic signal of the input signal x.
    
    Returns:
    --------
    tfd : ndarray
        Time-lag distribution of the analytic signal z.

    """
    N = len(z) // 2  # Assuming z is a signal of length 2N
    Nh = int(np.ceil(N / 2))

    # Initialize the time-frequency distribution (TFD) matrix
    tfd = np.zeros((N, N), dtype=complex)

    m = np.arange(Nh)
    
    # Loop over time indices
    for n in range(N):
        inp = np.mod(n + m, 2 * N)
        inn = np.mod(n - m, 2 * N)

        # Extract the time slice from the analytic signal
        K_time_slice = z[inp] * np.conj(z[inn])

        # Store real and imaginary parts
        tfd[n, :Nh] = np.real(K_time_slice)
        tfd[n, Nh:] = np.imag(K_time_slice)
    
    return tfd

def multiply_kernel_signal(tfd, kern_type, kern_params, N, Nh):
    """
    Multiplies the TFD by the Doppler-lag kernel.

    Parameters:
    -----------
    tfd : ndarray
        Time-frequency distribution.
    kern_type : str
        Kernel type to be applied.
    kern_params : list
        Kernel parameters specific to the kernel type.
    N : int
        Length of the signal.
    Nh : int
        Half length of the signal.

    Returns:
    --------
    tfd : ndarray
        Modified TFD after kernel multiplication.
    """
    # Loop over lag indices
    for m in range(Nh):
        # Generate the Doppler-lag kernel for each lag index
        g_lag_slice = gen_doppler_lag_kern(kern_type, kern_params, N, m)
        
        # Extract and transform the TFD slice for this lag
        tfd_slice = np.fft.fft(tfd[:, m]) + 1j * np.fft.fft(tfd[:, Nh + m])
        
        # Multiply by the kernel and perform inverse FFT
        R_lag_slice = np.fft.ifft(tfd_slice * g_lag_slice)
        
        # Store real and imaginary parts back into the TFD
        tfd[:, m] = np.real(R_lag_slice)
        tfd[:, Nh + m] = np.imag(R_lag_slice)
    
    return tfd

def gen_doppler_lag_kern(kern_type, kern_params, N, lag_index):
    """
    Generate the Doppler-lag kernel based on kernel type and parameters.

    Parameters:
    -----------
    kern_type : str
        Type of kernel (e.g., 'wvd', 'swvd', 'pwvd', etc.).
    kern_params : list
        Parameters for the kernel.
    N : int
        Signal length.
    lag_index : int
        Current lag index.

    Returns:
    --------
    g : ndarray
        Doppler-lag kernel for the given lag.
    """
    g = np.zeros(N, dtype=complex)  # Initialize the kernel

    if not isinstance(kern_params, list):
        kern_params = [kern_params]  # Ensure parameters are in a list

    # Get kernel based on the type
    g = get_kern(g, lag_index, kern_type, kern_params, N)

    return np.real(g) # All kernels are real valued


def get_kern(g, lag_index, kernel_type, kernel_params, N):
    """
    Get the kernel based on the provided kernel type.

    Parameters:
    -----------
    g : ndarray
        Kernel to be filled.
    lag_index : int
        Lag index.
    kernel_type : str
        Type of kernel ('wvd', 'swvd', 'pwvd', 'sep').
    kernel_params : list
        Parameters for the kernel.
    N : int
        Signal length.

    Returns:
    --------
    g : ndarray
        Kernel function at the current lag.
    """
    l = len(kernel_params)

    if kernel_type == 'wvd':
        g[:] = 1 # WVD kernel is the equal to 1 for all lags

    elif kernel_type == 'swvd':
        # Smoothed Wigner-Ville Distribution (Lag Independent kernel)
        if l < 2:
            raise ValueError("Need at least two window parameters for SWVD")
        win_length = kernel_params[0]
        win_type = kernel_params[1]
        win_param = kernel_params[2] if l >= 3 else 0
        win_param2 = kernel_params[3] if l >= 4 else 1

        G1 = get_window(win_length, win_type, win_param)
        G1 = pad_window(G1, N)

        # Define window in the time domain or Doppler domain
        if win_param2 == 0:
            G1 = np.fft.fft(G1)
            G1 /= G1[0]

        g[:] = G1 # Assign the window to the kernel

    elif kernel_type == 'pwvd':
        # Pseudo-Wigner-Ville Distribution (Doppler Independent kernel)
        if l < 2:
            raise ValueError("Need at least two window parameters for PWVD") 
        win_length = kernel_params[0]  
        win_type = kernel_params[1] 
        win_param = kernel_params[2] if l >= 3 else 0
        win_param2 = kernel_params[3] if l >= 4 else 0

        G2 = get_window(win_length, win_type, win_param) # Generate the window, same per lag iteration
        G2 = pad_window(G2, N)      # Zero-pad the window to the length of the signal
        G2 = G2[lag_index]          # Extract the lag_index-th element of the window --> can this be done more efficiently? Since we only need one element of the window and the padding is similar for every iteration

        g[:] = G2 # Assign the window to the kernel

    elif kernel_type == 'sep':
        # Separable Kernel
        g1 = np.copy(g)  # Create a new array for g1
        g2 = np.copy(g)  # Create a new array for g2
        
        # Call recursively to obtain g1 and g2 kernels (no in-place modification of g)
        g1 = get_kern(g1, lag_index, 'swvd', kernel_params[0], N) # Generate the first kernel
        g2 = get_kern(g2, lag_index, 'pwvd', kernel_params[1], N) # Generate the second kernel
        g = g1 * g2 # Multiply the two kernels to obtain the separable kernel

    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    return g

def get_window(win_length, win_type, win_param=None, dft_window=False, Npad=0):
    """
    General function to calculate a window function.
    
    Parameters:
    -----------
    win_length : int
        Length of the window.
    win_type : str
        Type of window. Options are:
        {'delta', 'rect', 'hamm'|'hamming', 'hann'|'hanning', 'gauss', 'cosh'}.
    win_param : float, optional
        Window parameter (e.g., alpha for Gaussian window). Default is None.
    dft_window : bool, optional
        If True, returns the DFT of the window. Default is False.
    Npad : int, optional
        If greater than 0, zero-pads the window to length Npad. Default is 0.
    
    Returns:
    --------
    win : ndarray
        The calculated window (or its DFT if dft_window is True).
    """
    # Handle optional arguments
    if win_param is None:
        win_param = []
    
    # Get the window
    win = get_win(win_length, win_type, win_param, dft_window)
    
    # Shift the window so that positive indices are first
    win = shift_window(win)
    
    # Zero-pad the window to length Npad if necessary
    if Npad > 0:
        win = pad_window(win, Npad)
    
    return win

def get_win(win_length, win_type, win_param=None, dft_window=False):
    """
    Helper function to create the specified window type.
    
    Parameters:
    -----------
    win_length : int
        Length of the window.
    win_type : str
        Type of window.
    win_param : float, optional
        Additional parameter for certain window types (e.g., Gaussian alpha). Default is None.
    dft_window : bool, optional
        If True, returns the DFT of the window. Default is False.
    
    Returns:
    --------
    win : ndarray
        The created window (or its DFT if dft_window is True).
    """
    if win_type == 'delta':
        win = np.zeros(win_length)
        win[win_length // 2] = 1
    elif win_type == 'rect':
        win = np.ones(win_length)
    elif win_type in ['hamm', 'hamming']:
        win = signal.windows.hamming(win_length)
    elif win_type in ['hann', 'hanning']:
        win = signal.windows.hann(win_length)
    elif win_type == 'gauss':
        win = signal.windows.gaussian(win_length, std=win_param if win_param else 0.4)
    elif win_type == 'cosh':
        win_hlf = win_length // 2
        if not win_param:
            win_param = 0.01
        win = np.array([np.cosh(m) ** (-2 * win_param) for m in range(-win_hlf, win_hlf+1)])
        win = np.fft.fftshift(win)
    else:
        raise ValueError(f"Unknown window type {win_type}")
    
    # If dft_window is True, return the DFT of the window
    if dft_window:
        win = np.fft.fft(np.roll(win, win_length // 2))
        win = np.roll(win, -win_length // 2)
    
    return win

def shift_window(w):
    """
    Shift the window so that positive indices appear first.
    
    Parameters:
    -----------
    w : ndarray
        Window to be shifted.
    
    Returns:
    --------
    w_shifted : ndarray
        Shifted window with positive indices first.
    """
    N = len(w)
    return np.roll(w, N // 2)

def pad_window(w, Npad):
    """
    Zero-pad the window to a specified length.
    
    Parameters:
    -----------
    w : ndarray
        The original window.
    Npad : int
        Length to zero-pad the window to.
    
    Returns:
    --------
    w_pad : ndarray
        Zero-padded window of length Npad.
    
    Raises:
    -------
    ValueError:
        If Npad is less than the original window length.
    """
    N = len(w)
    w_pad = np.zeros(Npad)
    Nh = N // 2
    
    if Npad < N:
        raise ValueError("Npad must be greater than or equal to the window length")

    if N == Npad:
        return w
    
    if N % 2 == 1:  # For odd N
        w_pad[:Nh+1] = w[:Nh+1]
        w_pad[-Nh:] = w[-Nh:]
    else:  # For even N
        w_pad[:Nh] = w[:Nh]
        w_pad[Nh] = w[Nh] / 2
        w_pad[-Nh:] = w[-Nh:]
        w_pad[-Nh] = w[Nh] / 2
    
    return w_pad

def compute_tfd(N, Nh, tfd):
    """
    Finalizes the time-frequency distribution computation.

    Parameters:
    -----------
    N : int
        Size of the TFD.
    Nh : int
        Half-length parameter.
    tfd : ndarray
        Time-frequency distribution to be finalized.

    Returns:
    --------
    tfd : ndarray
        Final computed TFD (N,N).
    """
    m = np.arange(0, Nh)  # m = 0:(Nh-1)
    mb = np.arange(1, Nh)  # mb = 1:(Nh-1)

    m_real = np.arange(Nh) # m_real = 0:(Nh-1) --> is this the same as m? In matlab it is different because of 1-based indexing
    m_imag = np.arange(Nh, N) # m_imag = Nh:(N-1)

    for n in range(0, N-1, 2):  # for n=0:2:N-2
        R_even_half = np.complex128(tfd[n, :Nh]) + 1j * np.complex128(tfd[n, Nh:])
        R_odd_half = np.complex128(tfd[n+1, :Nh]) + 1j * np.complex128(tfd[n+1, Nh:])

        R_tslice_even = np.zeros(N, dtype=np.complex128)  
        R_tslice_odd = np.zeros(N, dtype=np.complex128)

        R_tslice_even[m] = R_even_half
        R_tslice_odd[m] = R_odd_half

        R_tslice_even[N-mb] = np.conj(R_even_half[mb])  
        R_tslice_odd[N-mb] = np.conj(R_odd_half[mb])
        
        # Perform FFT to compute time slices
        tfd_time_slice = np.fft.fft(R_tslice_even + 1j * R_tslice_odd)

        tfd[n, :] = np.real(tfd_time_slice)
        tfd[n+1, :] = np.imag(tfd_time_slice)

    tfd = tfd / N  # Normalize the TFD
    tfd = tfd.transpose()  # Transpose the TFD to have the time on the x-axis and frequency on the y-axis   
    return tfd


