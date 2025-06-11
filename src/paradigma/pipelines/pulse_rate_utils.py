import numpy as np
from scipy import signal
from typing import Tuple

from paradigma.config import PulseRateConfig


def assign_sqa_label(
        ppg_prob: np.ndarray, 
        config: PulseRateConfig, 
        acc_label=None
    ) -> np.ndarray:
    """
    Assigns a signal quality label to every individual data point.

    Parameters
    ----------
    ppg_prob : np.ndarray
        The probabilities for PPG.
    config : PulseRateConfig
        The configuration parameters.
    acc_label : np.ndarray, optional
        The labels for the accelerometer.

    Returns
    -------
    np.ndarray
        The signal quality assessment labels.
    """
    # Default _label to ones if not provided
    if acc_label is None:
        acc_label = np.ones(len(ppg_prob))

    # Number of samples in an epoch
    fs = config.sampling_frequency
    samples_per_epoch = config.window_length_s * fs

    # Calculate number of samples to shift for each epoch
    samples_shift = config.window_step_length_s * fs
    n_samples = int(np.round(len(ppg_prob) + config.window_overlap_s) * fs)
    data_prob = np.zeros(n_samples)
    data_label_imu = np.zeros(n_samples, dtype=np.int8)

    for i in range(n_samples):
        # Start and end indices for current epoch
        start_idx = max(0, int((i - (samples_per_epoch - samples_shift)) // fs)) # max to handle first epochs
        end_idx = min(int(i // fs), len(ppg_prob))  # min to handle last epochs  

        # Extract probabilities and labels for the current epoch
        prob = ppg_prob[start_idx:end_idx+1]
        label_imu = acc_label[start_idx:end_idx+1]

        # Calculate mean probability and majority voting for labels
        data_prob[i] = np.mean(prob)
        data_label_imu[i] = int(np.mean(label_imu) >= 0.5)

    # Set probability to zero if majority IMU label is 0
    data_prob[data_label_imu == 0] = 0
    sqa_label = data_prob >= config.threshold_sqa

    return sqa_label


def extract_pr_segments(sqa_label: np.ndarray, min_pr_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts pulse rate segments based on the SQA label.

    Parameters
    ----------
    sqa_label : np.ndarray
        The signal quality assessment label.
    min_pr_samples : int
        The minimum number of samples required for a pulse rate segment.

    Returns
    -------
    Tuple[v_start_idx_long, v_end_idx_long]
        The start and end indices of the pulse rate segments.
    """
    # Find the start and end indices of the pulse rate segments
    v_start_idx = np.where(np.diff(sqa_label.astype(int)) == 1)[0] + 1
    v_end_idx = np.where(np.diff(sqa_label.astype(int)) == -1)[0] + 1

    # Check if the first segment is a start or end
    if sqa_label[0] == 1:
        v_start_idx = np.insert(v_start_idx, 0, 0)
    if sqa_label[-1] == 1:
        v_end_idx = np.append(v_end_idx, len(sqa_label))

    # Check if the segments are long enough
    v_start_idx_long = v_start_idx[(v_end_idx - v_start_idx) >= min_pr_samples]
    v_end_idx_long = v_end_idx[(v_end_idx - v_start_idx) >= min_pr_samples]

    return v_start_idx_long, v_end_idx_long


def extract_pr_from_segment(
        ppg: np.ndarray, 
        tfd_length: int, 
        fs: int, 
        kern_type: str, 
        kern_params: dict
    ) -> np.ndarray:
    """
    Extracts pulse rate from the time-frequency distribution of the PPG signal.

    Parameters
    ----------
    ppg : np.ndarray
        The preprocessed PPG segment with 2 seconds of padding on both sides to reduce boundary effects.
    tfd_length : int
        Length of each segment (in seconds) to calculate the time-frequency distribution.
    fs : int
        The sampling frequency of the PPG signal.
    kern_type : str
        Type of TFD kernel to use (e.g., 'wvd' for Wigner-Ville distribution).
    kern_params : dict
        Parameters for the specified kernel. Not required for 'wvd', but relevant for other 
        kernels like 'spwvd' or 'swvd'. Default is None.

    Returns
    -------
    np.ndarray
        The estimated pulse rate.
    """

    # Constants to handle boundary effects
    edge_padding = 4 * fs  # Additional 4 seconds (2 seconds on both sides)
    tfd_length = tfd_length * fs  # Convert tfd_length to samples

    # Calculate the actual segment length without padding
    original_segment_length = len(ppg) - edge_padding

    # Determine the number of tfd_length-second segments
    num_segments = max(1, int(original_segment_length // tfd_length))

    # Split the PPG signal into segments
    ppg_segments = []
    for i in range(num_segments):
        if i != num_segments - 1:  # For all segments except the last
            start_idx = int(i * tfd_length)
            end_idx = int((i + 1) * tfd_length + edge_padding)
        else:  # For the last segment
            start_idx = int(i * tfd_length)
            end_idx = len(ppg)
        ppg_segments.append(ppg[start_idx:end_idx])

    pr_est_from_ppg = np.array([])
    for segment in ppg_segments:
        # Calculate the time-frequency distribution
        pr_tfd = extract_pr_with_tfd(segment, fs, kern_type, kern_params)
        pr_est_from_ppg = np.concatenate((pr_est_from_ppg, pr_tfd)) 

    return pr_est_from_ppg


def extract_pr_with_tfd(
        ppg: np.ndarray, 
        fs: int, 
        kern_type: str, 
        kern_params: dict
    ) -> np.ndarray:
    """
    Estimate pulse rate (PR) from a PPG segment using a TFD method with optional 
    moving average filtering.

    Parameters
    ----------
    ppg_segment : array-like
        Segment of the PPG signal to process.
    fs : int
        Sampling frequency in Hz.
    kern_type : str
        Type of TFD kernel to use.
    kern_params : dict
        Parameters for the specified kernel.

    Returns
    -------
    pr_smooth_tfd : np.ndarray
        Estimated pr values (in beats per minute) for each 2-second segment of the PPG signal.
    """
    # Generate the TFD matrix using the specified kernel
    tfd_obj = TimeFreqDistr()
    tfd = tfd_obj.nonsep_gdtfd(ppg, kern_type, kern_params)  # Returns an NxN matrix

    # Get time and frequency axes for the TFD
    num_time_samples, num_freq_bins = tfd.shape
    time_axis = np.arange(num_time_samples) / fs
    freq_axis = np.linspace(0, 0.5, num_freq_bins) * fs

    # Estimate pulse rate by identifying the max frequency in the TFD
    max_freq_indices = np.argmax(tfd, axis=0)

    pr_smooth_tfd = np.array([])
    for i in range(2, int(len(ppg) / fs) - 4 + 1, 2):  # Skip the first and last 2 seconds, add 1 to include the last segment
        relevant_indices = (time_axis >= i) & (time_axis < i + 2)
        avg_frequency = np.mean(freq_axis[max_freq_indices[relevant_indices]])
        pr_smooth_tfd = np.concatenate((pr_smooth_tfd, [60 * avg_frequency]))  # Convert frequency to BPM

    return pr_smooth_tfd


class TimeFreqDistr:
    def __init__(self):
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
        pass

    def nonsep_gdtfd(
            self,
            x: np.ndarray, 
            kern_type: None | str = None,
            kern_params: None | dict = None
        ):
        """
        Computes the generalized time-frequency distribution (TFD) using a non-separable kernel.

        Parameters:
        -----------
        x : ndarray
            Input signal to be analyzed.
        kern_type : str, optional
            Type of kernel to be used for TFD computation. Default is None.
            Currently supported kernels are:
                wvd  - kernel for Wigner-Ville distribution
                swvd - kernel for smoothed Wigner-Ville distribution
                    (lag-independent kernel)
                pwvd - kernel for pseudo Wigner-Ville distribution
                    (Doppler-independent kernel)
                sep  - kernel for separable kernel (combintation of SWVD and PWVD)

        kern_params : dict, optional
            Dictionary of parameters specific to the kernel type. Default is None.
            The structure of the dictionary depends on the selected kernel type:
                - wvd:
                An empty dictionary, as no additional parameters are required.
                - swvd:
                Dictionary with the following keys:
                    'win_length': Length of the smoothing window.
                    'win_type': Type of window function (e.g., 'hamm', 'hann').
                    'win_param' (optional): Additional parameters for the window.
                    'win_param2' (optional): 0 for time-domain window or 1 for Doppler-domain window.

            Example:
            ```python
            kern_params = {
                'win_length': 11,
                'win_type': 'hamm',
                'win_param': 0,
                'domain': 1
            }
            ```
            - pwvd:
            Dictionary with the following keys:
                'win_length': Length of the smoothing window.
                'win_type': Type of window function (e.g., 'cosh').
                'win_param' (optional): Additional parameters for the window.
                'win_param2' (optional): 0 for time-domain window or 1 for Doppler-domain window.
            Example:
            ```python
            kern_params = {
                'win_length': 200,
                'win_type': 'cosh',
                'win_param': 0.1
            }
            ```
            - sep:
            Dictionary containing two nested dictionaries, one for the Doppler window and one for the lag window:
                'doppler': {
                    'win_length': Length of the Doppler-domain window.
                    'win_type': Type of Doppler-domain window function.
                    'win_param' (optional): Additional parameters for the Doppler window.
                    'win_param2' (optional): 0 for time-domain window or 1 for Doppler-domain window.
                }
                'lag': {
                    'win_length': Length of the lag-domain window.
                    'win_type': Type of lag-domain window function.
                    'win_param' (optional): Additional parameters for the lag window.
                    'win_param2' (optional): 0 for time-domain window or 1 for Doppler-domain window.
                }
            Example:
            ```python
            kern_params = {
                'doppler': {
                    'win_length': doppler_samples,
                    'win_type': win_type_doppler,
                },
                'lag': {
                    'win_length': lag_samples,
                    'win_type': win_type_lag,
                }
            }
            ```

        Returns:
        --------
        tfd : ndarray
            The computed time-frequency distribution.
        """
        z = self.get_analytic_signal(x)
        N = len(z) // 2  # Since z is a signal of length 2N
        Nh = int(np.ceil(N / 2))

        # Generate the time-lag distribution of the analytic signal
        tfd = self.gen_time_lag(z)

        # Multiply the TFD by the Doppler-lag kernel
        tfd = self.multiply_kernel_signal(tfd, kern_type, kern_params, N, Nh)
        
        # Finalize the TFD computation
        tfd = self.compute_tfd(N, Nh, tfd)
        
        return tfd

    def get_analytic_signal(self, x: np.ndarray) -> np.ndarray:
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
        z = self.gen_analytic(x)  
        
        return z

    def gen_analytic(self, x: np.ndarray) -> np.ndarray:
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
        H = np.empty(2 * N) # Preallocate an array of size 2*N
        H[0] = 1 # First element
        H[1:N] = 2 # Next N-1 elements
        H[N] = 1 # Middle element
        H[N+1:] = 0 # Last N-1 elements
        z_cb = np.fft.ifft(x_fft * H)

        # Force the second half of the time-domain signal to zero
        z = np.concatenate((z_cb[:N], np.zeros(N)))

        return z

    def gen_time_lag(self, z: np.ndarray) -> np.ndarray:
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

    def multiply_kernel_signal(  
        self,
        tfd: np.ndarray, 
        kern_type: str, 
        kern_params: dict, 
        N: int, 
        Nh: int  
        ) -> np.ndarray:  
        """
        Multiplies the TFD by the Doppler-lag kernel.

        Parameters:
        -----------
        tfd : ndarray
            Time-frequency distribution.
        kern_type : str
            Kernel type to be applied.
        kern_params : dict
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
            g_lag_slice = self.gen_doppler_lag_kern(kern_type, kern_params, N, m)
            
            # Extract and transform the TFD slice for this lag
            tfd_slice = np.fft.fft(tfd[:, m]) + 1j * np.fft.fft(tfd[:, Nh + m])
            
            # Multiply by the kernel and perform inverse FFT
            R_lag_slice = np.fft.ifft(tfd_slice * g_lag_slice)
            
            # Store real and imaginary parts back into the TFD
            tfd[:, m] = np.real(R_lag_slice)
            tfd[:, Nh + m] = np.imag(R_lag_slice)
        
        return tfd

    def gen_doppler_lag_kern(
            self, 
            kern_type: str, 
            kern_params: dict, 
            N: int, 
            lag_index: int
        ):
        """
        Generate the Doppler-lag kernel based on kernel type and parameters.

        Parameters:
        -----------
        kern_type : str
            Type of kernel (e.g., 'wvd', 'swvd', 'pwvd', etc.).
        kern_params : dict
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

        # Get kernel based on the type
        g = self.get_kern(g, lag_index, kern_type, kern_params, N)

        return np.real(g) # All kernels are real valued

    def get_kern(
            self, 
            g: np.ndarray, 
            lag_index: int, 
            kern_type: str, 
            kern_params: dict, 
            N: int
        ) -> np.ndarray:
        """
        Get the kernel based on the provided kernel type.

        Parameters:
        -----------
        g : ndarray
            Kernel to be filled.
        lag_index : int
            Lag index for the kernel.
        kern_type : str
            Type of kernel to use (now included: 'wvd', 'swvd', 'pwvd', 'sep').
        kern_params : dict
            Parameters for the specified kernel.
        N : int
            Signal length.

        Returns:
        --------
        g : ndarray
            Kernel function at the current lag.
        """
        # Validate kern_type
        valid_kern_types = ['wvd', 'sep', 'swvd', 'pwvd']  # List of valid kernel types which are currently supported
        if kern_type not in valid_kern_types:
            raise ValueError(f"Unknown kernel type: {kern_type}. Expected one of {valid_kern_types}")
        
        num_params = len(kern_params)

        if kern_type == 'wvd':
            g[:] = 1 # WVD kernel is the equal to 1 for all lags

        elif kern_type == 'sep':
            # Separable Kernel
            g1 = np.copy(g)  # Create a new array for g1
            g2 = np.copy(g)  # Create a new array for g2
            
            # Call recursively to obtain g1 and g2 kernels (no in-place modification of g)
            g1 = self.get_kern(g1, lag_index, 'swvd', kern_params['lag'], N) # Generate the first kernel
            g2 = self.get_kern(g2, lag_index, 'pwvd', kern_params['doppler'], N) # Generate the second kernel
            g = g1 * g2 # Multiply the two kernels to obtain the separable kernel

        else:
            if num_params < 2:
                raise ValueError("Missing required kernel parameters: 'win_length' and 'win_type'")

            win_length = kern_params['win_length']
            win_type = kern_params['win_type']
            win_param = kern_params['win_param'] if 'win_param' in kern_params else 0
            win_param2 = kern_params['win_param2'] if 'win_param2' in kern_params else 1

            G = self.get_window(win_length, win_type, win_param)
            G = self.pad_window(G, N)

            if kern_type == 'swvd' and win_param2 == 0:
                G = np.fft.fft(G)
                if G[0] != 0:  # add this check to avoid division by zero
                    G /= G[0]
                G = G[lag_index]

            g[:] = G

        return g

    def get_window(
            self, 
            win_length: int, 
            win_type: str,
            win_param: float | None = None,
            dft_window: bool = False,
            Npad: int = 0
        ) -> np.ndarray:
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
        
        # Get the window
        win = self.get_win(win_length, win_type, win_param, dft_window)
        
        # Shift the window so that positive indices are first
        win = self.shift_window(win)
        
        # Zero-pad the window to length Npad if necessary
        if Npad > 0:
            win = self.pad_window(win, Npad)
        
        return win

    def get_win(
            self, 
            win_length: int, 
            win_type: str, 
            win_param: float | None = None, 
            dft_window: bool = False
        ) -> np.ndarray:
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

    def shift_window(self, w: np.ndarray) -> np.ndarray:
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

    def pad_window(self, w: np.ndarray, Npad: int) -> np.ndarray:
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

    def compute_tfd(
            self, 
            N: int, 
            Nh: int, 
            tfd: np.ndarray
        ):
        """
        Finalizes the time-frequency distribution computation.

        Parameters:
        -----------
        N : int
            Size of the TFD.
        Nh : int
            Half-length parameter.
        tfd : np.ndarray
            Time-frequency distribution to be finalized.

        Returns:
        --------
        tfd : np.ndarray
            Final computed TFD (N,N).
        """
        m = np.arange(0, Nh)  # m = 0:(Nh-1)
        mb = np.arange(1, Nh)  # mb = 1:(Nh-1)

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
