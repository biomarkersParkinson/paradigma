import numpy as np
import pandas as pd
from scipy import signal
from paradigma.constants import DataColumns
from paradigma.gait.feature_extraction import compute_total_power

def melscale(x):
    "Maps values of x to the melscale"
    return 64.875 * np.log10(1 + x / 17.5)

def inverse_melscale(x):
    "Inverse of the melscale"
    return 17.5 * (10 ** (x / 64.875) - 1)
    
def compute_mfccs(
        config,
        total_power_array: np.ndarray,
        mel_scale: bool = True,
        ) -> np.ndarray:
    """
    Generate Mel Frequency Cepstral Coefficients (MFCCs) from the total power spectral density of the signal.

    MFCCs are commonly used features in signal processing for tasks like audio and 
    vibration analysis. In this version, we adjusted the MFFCs to the human activity
    range according to: https://www.sciencedirect.com/science/article/abs/pii/S016516841500331X#f0050.
    This function calculates MFCCs by applying a filterbank 
    (in either the mel scale or linear scale) to the total power spectral density of the signal, 
    followed by a Discrete Cosine Transform (DCT) to obtain coefficients.

    Parameters
    ----------
    config : object
        Configuration object containing the following attributes:
        - window_length_s : int
            Duration of each analysis window in seconds (default: 4 s).
        - sampling_frequency : int
            Sampling frequency of the data (default: 100 Hz).
        - mfcc_low_frequency : float
            Lower bound of the frequency band (default: 0 Hz).
        - mfcc_high_frequency : float
            Upper bound of the frequency band (default: 25 Hz).
        - mfcc_n_dct_filters : int
            Number of triangular filters in the filterbank (default: 15).
        - mfcc_n_coefficients : int
            Number of coefficients to extract (default: 12).
    total_power_array : np.ndarray
        2D array of shape (n_windows, n_frequencies) containing the total power 
        of the signal for each window.
    mel_scale : bool, optional
        Whether to use the mel scale for the filterbank (default: True).

    Returns
    -------
    np.ndarray
        2D array of MFCCs with shape `(n_windows, n_coefficients)`, where each row
        contains the MFCCs for a corresponding window.
    ...

    Raises
    ------
    ValueError
        If the filter points cannot be constructed due to incompatible dimensions.

    Notes
    -----
    - The function includes filterbank normalization to ensure proper scaling.
    - DCT filters are constructed to minimize spectral leakage.
    """

    # Compute window length in samples
    window_length = config.window_length_s * config.sampling_frequency
    
    # Generate filter points
    if mel_scale:
        freqs = np.linspace(
            melscale(config.mfcc_low_frequency), 
            melscale(config.mfcc_high_frequency), 
            num=config.mfcc_n_dct_filters + 2
        )
        freqs = inverse_melscale(freqs)
    else:
        freqs = np.linspace(
            config.mfcc_low_frequency, 
            config.mfcc_high_frequency, 
            num=config.mfcc_n_dct_filters + 2
        )

    filter_points = np.round(
        (window_length) / config.sampling_frequency * freqs 
    ).astype(int) + 1

    # Construct triangular filterbank
    filters = np.zeros((len(filter_points) - 2, int(window_length / 2 + 1)))
    for j in range(len(filter_points) - 2):
        filters[j, filter_points[j] : filter_points[j + 2]] = signal.windows.triang(
            filter_points[j + 2] - filter_points[j]
        ) 
        # Normalize filter coefficients
        filters[j, :] /= (
            config.sampling_frequency/window_length * np.sum(filters[j,:])
        ) 

    # Apply filterbank to total power
    power_filtered = np.dot(total_power_array, filters.T) 
    
    # Convert power to logarithmic scale
    log_power_filtered = np.log10(power_filtered + 1e-10)

    # Generate DCT filters
    dct_filters = np.empty((config.mfcc_n_coefficients, config.mfcc_n_dct_filters))
    dct_filters[0, :] = 1.0 / np.sqrt(config.mfcc_n_dct_filters)

    samples = (
        np.arange(1, 2 * config.mfcc_n_dct_filters, 2) * np.pi / (2.0 * config.mfcc_n_dct_filters)
    )

    for i in range(1, config.mfcc_n_coefficients):
        dct_filters[i, :] = np.cos(i * samples) * np.sqrt(2.0 / config.mfcc_n_dct_filters)

    # Compute MFCCs
    mfccs = np.dot(log_power_filtered, dct_filters.T) 

    return mfccs


def extract_frequency_peak(
    total_psd: np.ndarray,
    freq_vect: np.ndarray,
    fmin: float = 1,
    fmax: float = 25
    ) -> pd.Series:

    """Extract the frequency of the peak in the power spectral density within the specified frequency band.
    
    Parameters
    ----------
    total_psd: pd.Series
        The total power spectral density of the gyroscope signal
    freq_vect: pd.Series
        Frequency vector corresponding to the power spectral density
    fmin: float
        The lower bound of the frequency band in Hz (default: 1)
    fmax: float
        The upper bound of the frequency band in Hz (default: 25)
        
    Returns
    -------
    pd.Series
        The frequency of the peak across windows
    """    
    freq_idx = np.where((freq_vect>=fmin) & (freq_vect<=fmax))[0]
    peak_idx = np.argmax(total_psd[:, freq_idx], axis=1)
    frequency_peak = freq_vect[freq_idx][peak_idx]

    return frequency_peak


def extract_low_freq_power(
    total_psd: np.ndarray,
    freq_vect: np.ndarray,
    fmin: float = 0.5,
    fmax: float = 3,
    spectral_resolution: float = 0.25
    ) -> pd.Series:

    """Computes the power in the low frequency power range across windows (for slow arm movement detection).
    
    Parameters
    ----------
    total_psd: pd.Series
        The total power spectral density of the gyroscope signal
    freq_vect: pd.Series
        Frequency vector corresponding to the power spectral density
    fmin: float
        The lower bound of the frequency band in Hz (default: 0.5)
    fmax: float
        The upper bound of the frequency band in Hz (default: 3)
    spectral_resolution: float
        The spectral resolution of the PSD in Hz (default: 0.25)
        
    Returns
    -------
    pd.Series
        The power in the low frequency power range across windows
    """
    
    freq_idx = (freq_vect>=fmin) & (freq_vect<fmax)
    bandpower = spectral_resolution * np.sum(total_psd[:, freq_idx], axis=1)

    return bandpower


def extract_tremor_power(
    total_psd: np.ndarray,
    freq_vect: np.ndarray,
    fmin: float = 3,
    fmax: float = 7,
    spectral_resolution: float = 0.25
    ) -> np.ndarray:

    """Computes the tremor power (1.25 Hz around the peak within the tremor frequency band)
    
    Parameters
    ----------
    total_psd: pd.Series
        The total power spectral density of the gyroscope signal
    freq_vect: pd.Series
        Frequency vector corresponding to the power spectral density
    fmin: float
        The lower bound of the tremor frequency band in Hz (default: 3)
    fmax: float
        The upper bound of the tremor frequency band in Hz (default: 7)
    spectral_resolution: float
        The spectral resolution of the PSD in Hz (default: 0.25)
        
    Returns
    -------
    pd.Series
        The tremor power across windows
    """
    
    freq_idx = (freq_vect>=fmin) & (freq_vect<=fmax)
    peak_idx = np.argmax(total_psd[:, freq_idx], axis=1) + np.min(np.where(freq_idx)[0])
    left_idx =  np.maximum((peak_idx - 0.5 / spectral_resolution).astype(int), 0)
    right_idx = (peak_idx + 0.5 / spectral_resolution).astype(int)

    row_indices = np.arange(total_psd.shape[1])
    row_indices = np.tile(row_indices, (total_psd.shape[0], 1))
    left_idx = left_idx[:, None]
    right_idx = right_idx[:, None]

    mask = (row_indices >= left_idx) & (row_indices <= right_idx)

    tremor_power = spectral_resolution * np.sum(total_psd*mask, axis=1)

    return tremor_power


def extract_spectral_domain_features(config, data) -> pd.DataFrame:
    """
    Compute spectral domain features from the gyroscope data.

    This function computes Mel-frequency cepstral coefficients (MFCCs), the frequency of the peak, 
    the tremor power, and the non-tremor power based on the total power spectral density of the windowed gyroscope data.

    Parameters
    ----------
    config : object
        Configuration object containing settings such as sampling frequency, window type, 
        and MFCC parameters.
    data : numpy.ndarray
        A 2D numpy array where each row corresponds to a window of gyroscope data.

    Returns
    -------
    pd.DataFrame
        The feature dataframe containing the extracted spectral features, including 
        MFCCs, the frequency of the peak, the tremor power and non-tremor power for each window.
    """

    # Initialize a dictionary to hold the results
    feature_dict = {}

    # Initialize parameters
    sampling_frequency = config.sampling_frequency
    segment_length_s = config.segment_length_s
    overlap_fraction = config.overlap_fraction
    spectral_resolution = config.spectral_resolution
    window_type = 'hann'

    # Compute the power spectral density
    segment_length_n = sampling_frequency * segment_length_s
    overlap_n = segment_length_n * overlap_fraction
    window = signal.get_window(window_type, segment_length_n, fftbins=False)
    nfft = sampling_frequency / spectral_resolution

    freqs, psd = signal.welch(
        x=data, 
        fs=sampling_frequency, 
        window=window, 
        nperseg=segment_length_n,
        noverlap=overlap_n, 
        nfft=nfft, 
        detrend=False, 
        scaling='density',
        axis=1
    )

    # Compute total power in the PSD
    total_psd = compute_total_power(psd)

    # Compute the MFCC's
    config.mfcc_low_frequency = config.fmin_mfcc
    config.mfcc_high_frequency = config.fmax_mfcc
    config.mfcc_n_dct_filters = config.n_dct_filters_mfcc
    config.mfcc_n_coefficients = config.n_coefficients_mfcc

    mfccs = compute_mfccs(
        config,
        total_power_array=total_psd,
    )

    # Combine the MFCCs into the features DataFrame
    mfcc_colnames = [f'mfcc_{x}' for x in range(1, config.mfcc_n_coefficients + 1)]
    for i, colname in enumerate(mfcc_colnames):
        feature_dict[colname] = mfccs[:, i]

    # Compute the frequency of the peak, non-tremor power and tremor power
    feature_dict['freq_peak'] = extract_frequency_peak(total_psd, freqs, config.fmin_peak, config.fmax_peak)
    feature_dict['low_freq_power'] = extract_low_freq_power(total_psd, freqs, config.fmin_low_power, config.fmax_low_power)
    feature_dict['tremor_power'] = extract_tremor_power(total_psd, freqs, config.fmin_tremor_power, config.fmax_tremor_power)

    return pd.DataFrame(feature_dict)
    