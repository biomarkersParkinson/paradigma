import numpy as np
import pandas as pd
from scipy import signal
from paradigma.constants import DataColumns
from paradigma.gait.feature_extraction import compute_mfccs

def compute_welch_periodogram(
        values: list,
        window_type: str = 'hann',
        sampling_frequency: int = 100,
        segment_length_s: float = 3,
        overlap_fraction: float = 0.8,
        spectral_resolution: float = 0.25
    )-> tuple:
    """Estimate power spectral density of the gyroscope signal using Welch's method.
    
    Parameters
    ----------
    values: List
        The values of the signal (e.g., gyroscope data) of a single window.
    window_type: str
        The type of window to be used for the PSD (default: 'hann')
    sampling_frequency: int
        The sampling frequency of the signal (default: 100)
    segment_length_s: float
        The length of each segment in seconds (default: 3)
    overlap_fraction: float
        The overlap between segments as fraction (default: 0.8)
    spectral_resolution: float
        The spectral resolution of the PSD in Hz (default: 0.25)
        
    Returns
    -------
    tuple
        Lists of PSD values and the corresponding frequencies
    """

    segment_length_n = sampling_frequency * segment_length_s
    overlap_n = segment_length_n * overlap_fraction
    window = signal.get_window(window_type, segment_length_n, fftbins=False)
    nfft = sampling_frequency / spectral_resolution
    
    f, Pxx = signal.welch(x=values, fs=sampling_frequency, window=window, nperseg=segment_length_n,
                          noverlap=overlap_n, nfft=nfft, detrend=False, scaling='density')

    return f, Pxx

def compute_welch_periodogram_numpy(
        values: np.ndarray,
        window: np.ndarray,
        segment_length_n: int,
        overlap_n: int,
        nfft: int,
        sampling_frequency: int = 100,
    )-> tuple:
    """Estimate power spectral density of the gyroscope signal using Welch's method.
    
    Parameters
    ----------
    values: np.ndarray
        The values of the signal (e.g., gyroscope data) of a single window.
    sampling_frequency: int
        The sampling frequency of the signal (default: 100)
    window: np.ndarray
        The window to be used for the PSD
    segment_length_n: int
        The length of each segment in samples
    overlap_n: int
        The overlap between segments in samples
    nfft: int
        The number of points to compute the FFT

    Returns
    -------
    tuple
        Lists of PSD values and the corresponding frequencies
    """
    
    f, Pxx = signal.welch(x=values, fs=sampling_frequency, window=window, nperseg=segment_length_n,
                          noverlap=overlap_n, nfft=nfft, detrend=False, scaling='density')

    return f, Pxx


def signal_to_PSD(
        sensor_col: pd.Series,
        window_type: str = 'hann',
        sampling_frequency: int = 100,
        segment_length_s: float = 3,
        overlap_fraction: float = 0.8,
        spectral_resolution: float = 0.25
    ) -> tuple:
    """Estimate the power spectral density (Welch's method) of a signal per window.

    Parameters
    ----------
    sensor_col: pd.Series
        The sensor column to be transformed (e.g. x-axis of gyroscope)
    window_type: str
        The type of window to be used for the PSD (default: 'hann')
    sampling_frequency: int
        The sampling frequency of the signal (default: 100)
    segment_length_s: float
        The length of each segment in seconds (default: 3)
    overlap_fraction: float
        The overlap between segments as fraction (default: 0.8)
    spectral_resolution: float
        The spectral resolution of the PSD in Hz (default: 0.25)
    
    Returns
    -------
    tuple
        Lists of PSD values and corresponding frequencies which can be concatenated as column to the dataframe
    """
    values_total = []
    freqs_total = []
    for row in sensor_col:
        freqs, values = compute_welch_periodogram(
            values=row,
            window_type=window_type,
            sampling_frequency=sampling_frequency,
            segment_length_s = segment_length_s,
            overlap_fraction = overlap_fraction,
            spectral_resolution = spectral_resolution)
        values_total.append(values)
        freqs_total.append(freqs)

    return freqs_total, values_total


def signal_to_PSD_numpy(
        sensor_data: np.ndarray,
        window_type: str = 'hann',
        sampling_frequency: int = 100,
        segment_length_s: float = 3,
        overlap_fraction: float = 0.8,
        spectral_resolution: float = 0.25
    ) -> tuple:
    """Estimate the power spectral density (Welch's method) of a signal per window.

    Parameters
    ----------
    sensor_data: np.ndarray
        The sensor data to be transformed (one or multiple axes of gyroscope)
    window_type: str
        The type of window to be used for the PSD (default: 'hann')
    sampling_frequency: int
        The sampling frequency of the signal (default: 100)
    segment_length_s: float
        The length of each segment in seconds (default: 3)
    overlap_fraction: float
        The overlap between segments as fraction (default: 0.8)
    spectral_resolution: float
        The spectral resolution of the PSD in Hz (default: 0.25)
    
    Returns
    -------
    tuple
        Lists of PSD values and corresponding frequencies which can be concatenated as column to the dataframe
    """

    values_total = []
    freqs_total = []

    # This was previously inside the for loop, but it is the same for all rows
    # if I'm correct, so I moved it outside the loop
    segment_length_n = sampling_frequency * segment_length_s
    overlap_n = segment_length_n * overlap_fraction
    window = signal.get_window(window_type, segment_length_n, fftbins=False)
    nfft = sampling_frequency / spectral_resolution

    for row_idx in range(sensor_data.shape[0]):
        freqs, values = compute_welch_periodogram_numpy(
            values=sensor_data[row_idx, :, :],
            sampling_frequency=sampling_frequency,
            window=window,
            segment_length_n=segment_length_n,
            overlap_n=overlap_n,
            nfft=nfft
        )
        
        values_total.append(values)
        freqs_total.append(freqs)

    return freqs_total, values_total

def compute_spectrogram(
        values: list,
        window_type: str = 'hann',
        sampling_frequency: int = 100,
        segment_length_s: float = 2,
        overlap_fraction: float = 0.8,
    )-> tuple:
    
    """Compute the spectrogram (using short time fourier transform) of the gyroscope signal
    
    Parameters
    ----------
    values: List
        The values of the signal (e.g., gyroscope data) of a single window.
    window_type: str
        The type of window to be used for the spectrogram (default: 'hann')
    sampling_frequency: int
        The sampling frequency of the signal (default: 100)
    segment_length_s: float
        The length of each segment in seconds (default: 2)
    overlap_fraction: float
        The overlap between segments as fraction (default: 0.8)
        
    Returns
    -------
    tuple
        Lists of spectrogram values
    """

    segment_length_n = sampling_frequency*segment_length_s
    overlap_n = segment_length_n*overlap_fraction
    window = signal.get_window(window_type,segment_length_n)

    f, t, S1 = signal.stft(x=values, fs=sampling_frequency, window=window, nperseg=segment_length_n, 
                           noverlap=overlap_n,boundary=None)

    return np.abs(S1)*sampling_frequency 

def signal_to_spectrogram(
        sensor_col: pd.Series,
        window_type: str = 'hann',
        sampling_frequency: int = 100,
        segment_length_s: float = 2,
        overlap_fraction: float = 0.8,
    ) -> list:
    
    """Compute the spectrogram (using short time fourier transform) of a signal per window.

    Parameters
    ----------
    sensor_col: pd.Series
        The sensor column to be transformed (e.g. x-axis of gyroscope)
    window_type: str
        The type of window to be used for the spectrogram (default: 'hann')
    sampling_frequency: int
        The sampling frequency of the signal (default: 100)
    segment_length_s: float
        The length of each segment in seconds (default: 2)
    overlap_fraction: float
        The overlap between segments as fraction (default: 0.8)
    
    Returns
    -------
    tuple
        Lists of spectrogram values which can be concatenated as column to the dataframe
    """
    spectrogram = []
    for row in sensor_col:
        spectrogram_values = compute_spectrogram(
            values=row,
            window_type=window_type,
            sampling_frequency=sampling_frequency,
            segment_length_s = segment_length_s,
            overlap_fraction = overlap_fraction
            )
        spectrogram.append(spectrogram_values)

    return spectrogram

def melscale(x):
    "Maps values of x to the melscale"
    return 64.875 * np.log10(1 + x / 17.5)

def inverse_melscale(x):
    "Inverse of the melscale"
    return 17.5 * (10 ** (x / 64.875) - 1)
    
def generate_mel_frequency_cepstral_coefficients(
        spectrogram: pd.Series,
        segment_length_s: float = 2,
        sampling_frequency: int = 100,
        fmin: float = 0,
        fmax: float = 25,
        n_filters: int = 15,
        n_coefficients: int = 12,
        ) -> pd.DataFrame:
    """Generate mel-frequency cepstral coefficients from the total spectrogram of the gyroscope signal.
    
    Parameters
    ----------
    spectrogram: pd.Series
        The total spectrogram of the gyroscope signal, extracted using stft
    segment_length_s: float
        The number of seconds a segment constitutes (default: 2)
    sampling_frequency: int
        The sampling frequency of the data (default: 100)
    fmin: float
        The lower bound of the frequency band (default: 0)
    fmax: float
        The upper bound of the frequency band (default: 25)
    n_filters: int
        The number of filters (default: 15)
    n_coefficients: int
        The number of coefficients to extract (default: 12)
    
    Returns
    -------
    pd.DataFrame
        A dataframe with a single column corresponding to a single mel-frequency cepstral coefficient
    """
    cepstral_coefs_total = []
    
    # construct mel-scale filters
    freqs = np.linspace(melscale(fmin), melscale(fmax), num=n_filters+2) # equal intervals in mel scale
    freqs_melscale = inverse_melscale(freqs) # convert to frequency domain
    segment_length = segment_length_s * sampling_frequency 
    filter_points = np.round((segment_length / sampling_frequency * freqs_melscale)).astype(int) + 1 # rounding the filter edges

    filters = np.zeros((len(filter_points)-2, int(segment_length/2+1)))
    for j in range(len(filter_points)-2):
        filters[j, filter_points[j] : filter_points[j+2]] = signal.windows.triang(filter_points[j+2] - filter_points[j]) # triangular filters based on edges
        filters[j,:] /= (sampling_frequency/segment_length * np.sum(filters[j,:])) # normalization of the filter coefficients
    
    # construct discrete cosine transform filters
    dct_filters = np.empty((n_coefficients, n_filters))
    dct_filters[0, :] = 1.0 / np.sqrt(n_filters)
    samples = np.arange(1, 2 * n_filters, 2) * np.pi / (2.0 * n_filters)
    for i in range(1, n_coefficients):
        dct_filters[i, :] = np.cos(i * samples) * np.sqrt(2.0 / n_filters)

    for spectrogram_window in spectrogram:
        # mel-filtering
        power_filtered = np.array([np.dot(filters, spectrogram_window[:, i]) for i in range(spectrogram_window.shape[1])])
        
        # taking the logarithm
        log_power_filtered = np.log10(power_filtered)

        # generate cepstral coefficients
        cepstral_coefs = np.array([np.dot(dct_filters, x) for x in log_power_filtered])

        # average coefficients over segments in window
        cepstral_coefs_total.append(np.transpose(np.mean(cepstral_coefs,axis=0)))

    return pd.DataFrame(cepstral_coefs_total, columns=['mfcc_{}'.format(j+1) for j in range(n_coefficients)])

def extract_frequency_peak(
    total_psd: pd.Series,
    freq_vect: pd.Series,
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
    frequency_peak = []
    
    freq_range = freq_vect[0]
    freq_idx = np.where((freq_range>=fmin) & (freq_range<=fmax))
    for psd_window in total_psd: 
        peak_idx = np.argmax(psd_window[freq_idx])
        freq_peak = freq_range[peak_idx]+fmin
        frequency_peak.append(freq_peak)

    return frequency_peak

def extract_frequency_peak_numpy(
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
    freq_idx = np.where((freq_vect>=fmin) & (freq_vect<=fmax))
    peak_idx = np.argmax(total_psd[:, freq_idx], axis=1)
    frequency_peak = freq_vect[peak_idx] + fmin

    return frequency_peak

def extract_low_freq_power(
    total_psd: pd.Series,
    freq_vect: pd.Series,
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

    low_freq_power = []
    
    freq_range = freq_vect[0]
    freq_idx = np.where((freq_range>=fmin) & (freq_range<fmax))
    for psd_window in total_psd: 
        bandpower = spectral_resolution*np.sum(psd_window[freq_idx])
        low_freq_power.append(bandpower)

    return low_freq_power

def extract_low_freq_power_numpy(
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
    total_psd: pd.Series,
    freq_vect: pd.Series,
    fmin: float = 3,
    fmax: float = 7,
    spectral_resolution: float = 0.25
    ) -> pd.Series:

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

    tremor_power = []
    
    freq_range = freq_vect[0]
    freq_idx = np.where((freq_range>=fmin) & (freq_range<=fmax))
    for psd_window in total_psd: 
        peak_idx = np.argmax(psd_window[freq_idx]) + np.min(freq_idx)
        left_idx =  np.max([0,int(peak_idx - 0.5/spectral_resolution)])
        right_idx = int(peak_idx + 0.5/spectral_resolution)
        peak_power = spectral_resolution*np.sum(psd_window[left_idx:right_idx+1])
        tremor_power.append(peak_power)

    return tremor_power

def extract_tremor_power_numpy(
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

    row_indices = np.arange(total_psd.shape[0])[:, None]
    left_idx = left_idx[:, None]
    right_idx = right_idx[:, None]

    mask = (row_indices >= left_idx) & (row_indices <= right_idx)

    tremor_power = spectral_resolution * np.sum(total_psd * mask, axis=0)

    return tremor_power

def extract_spectral_domain_features(config, df_windowed):

    # transform the temporal signal to the spectral domain using Welch's method and short time fourier transform
    for col in config.gyroscope_cols:
        df_windowed[f'{col}_freqs_PSD'], df_windowed[f'{col}_PSD'] = signal_to_PSD(
            sensor_col = df_windowed[col], 
            sampling_frequency = config.sampling_frequency,
            window_type = config.window_type, 
            segment_length_s = config.segment_length_s_psd,
            overlap_fraction = config.overlap_fraction,
            spectral_resolution = config.spectral_resolution_psd
            )
        df_windowed[f'{col}_spectrogram'] = signal_to_spectrogram(
            sensor_col = df_windowed[col], 
            sampling_frequency = config.sampling_frequency,
            window_type = config.window_type, 
            segment_length_s = config.segment_length_s_mfcc,
            overlap_fraction = config.overlap_fraction
            )
    
    # compute the total PSD and spectrogram (summed across the 3 gyroscope axes)
    df_windowed['total_PSD'] = df_windowed[[f"{y}_PSD" for y in config.gyroscope_cols]].sum(axis=1)
    df_windowed['total_spectrogram'] = df_windowed[[f"{y}_spectrogram"for y in config.gyroscope_cols]].sum(axis=1) 

    # compute the cepstral coefficients
    mfcc_cols = generate_mel_frequency_cepstral_coefficients(
        spectrogram=df_windowed['total_spectrogram'],
        segment_length_s = config.segment_length_s_mfcc,
        sampling_frequency=config.sampling_frequency,
        fmin=config.fmin_mfcc,
        fmax=config.fmax_mfcc,
        n_filters=config.n_dct_filters_mfcc,
        n_coefficients=config.n_coefficients_mfcc
        )
    df_windowed = pd.concat([df_windowed, mfcc_cols], axis=1)

    # compute the frequency of the peak in the PSD
    df_windowed['freq_peak'] = extract_frequency_peak(
        total_psd = df_windowed['total_PSD'],
        freq_vect = df_windowed['gyroscope_x_freqs_PSD'],
        fmin = config.fmin_peak,
        fmax = config.fmax_peak
    )
    
    # compute the low frequency power (for detection of slow arm movement)
    df_windowed['low_freq_power'] = extract_low_freq_power(
        total_psd = df_windowed['total_PSD'],
        freq_vect = df_windowed['gyroscope_x_freqs_PSD'],
        fmin = config.fmin_low_power,
        fmax = config.fmax_low_power,
        spectral_resolution = config.spectral_resolution_psd
    )

    # compute the tremor power
    df_windowed['tremor_power'] = extract_tremor_power(
        total_psd = df_windowed['total_PSD'],
        freq_vect = df_windowed['gyroscope_x_freqs_PSD'],
        fmin = config.fmin_tremor_power,
        fmax = config.fmax_tremor_power,
        spectral_resolution = config.spectral_resolution_psd 
    )

    df_windowed = df_windowed.rename(columns={'window_start': DataColumns.TIME})

    return df_windowed

def extract_spectral_domain_features_numpy(config, data):

    sampling_frequency = 100
    segment_length_s_psd = 3
    segment_length_s_mfcc = 2
    overlap_fraction = 0.8
    spectral_resolution = 0.25
    window_type = 'hann'

    segment_length_n = sampling_frequency * segment_length_s_psd
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

    segment_length_n = sampling_frequency * segment_length_s_mfcc
    overlap_n = segment_length_n * overlap_fraction
    window = signal.get_window(window_type, segment_length_n) # No FFTbins here?

    f, t, S1 = signal.stft(
        x=data, 
        fs=sampling_frequency, 
        window=window, 
        nperseg=segment_length_n, 
        noverlap=overlap_n,
        boundary=None,
        axis=1
    )

    total_psd = np.sum(psd, axis=2)
    total_spectrogram = np.sum(np.abs(S1), axis=2)

    config.mfcc_low_frequency = config.fmin_mfcc
    config.mfcc_high_frequency = config.fmax_mfcc
    config.mfcc_n_dct_filters = config.n_dct_filters_mfcc
    config.mfcc_n_coefficients = config.n_coefficients_mfcc

    mfccs = compute_mfccs(config, total_psd)

    df_features = pd.concat([df_features, pd.DataFrame(mfccs)], axis=1)

    d_spectral_features = {}
    d_spectral_features['freq_peak'] = extract_frequency_peak_numpy(total_psd, freqs, config.fmin_peak, config.fmax_peak)
    d_spectral_features['low_freq_power'] = extract_low_freq_power_numpy(total_psd, freqs, config.fmin_low_power, config.fmax_low_power)
    d_spectral_features['tremor_power'] = extract_tremor_power_numpy(total_psd, freqs, config.fmin_tremor_power, config.fmax_tremor_power)


    df_windowed = df_windowed.rename(columns={'window_start': DataColumns.TIME})

    return df_windowed
    