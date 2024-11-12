from typing import List
import numpy as np
import pandas as pd

from scipy import signal

from paradigma.constants import DataColumns

def compute_welch_periodogram(
        values: list,
        window_type = 'hann',
        sampling_frequency: int = 100,
        segment_length_s = 3,
        overlap: int = 0.8,
        spectral_resolution: int = 0.25
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
    segment_length_s: int
        The length of each segment in seconds
    overlap: int
        The overlap between segments 
    spectral_resolution: int
        The spectral resolution of the PSD in Hz
        
    Returns
    -------
    tuple
        Lists of PSD values and the corresponding frequencies
    """

    segment_length_n = sampling_frequency*segment_length_s
    overlap_n = segment_length_n*overlap
    window = signal.get_window(window_type, segment_length_n,fftbins=False)
    nfft = sampling_frequency/spectral_resolution
    
    f, Pxx = signal.welch(values,sampling_frequency,window,segment_length_n,
                          overlap_n,nfft,detrend=False,scaling='density')

    return f, Pxx

def signal_to_PSD(
        sensor_col: pd.Series,
        window_type = 'hann',
        sampling_frequency: int = 100,
        segment_length_s = 3,
        overlap: int = 0.8,
        spectral_resolution: int = 0.25
    ) -> tuple:
    """Compute the PSD (welch's method) of a signal per window.

    Parameters
    ----------
    sensor_col: pd.Series
        The sensor column to be transformed (e.g. x-axis of gyroscope)
    window_type: str
        The type of window to be used for the PSD (default: 'hann')
    sampling_frequency: int
        The sampling frequency of the signal (default: 100)
    segment_length_s: int
        The length of each segment in seconds
    overlap: int
        The overlap between segments
    spectral_resolution: int
        The spectral resolution of the PSD in Hz
    
    Returns
    -------
    tuple
        Lists of PSD values and corresponding frequencies which can be concatenated as column to the dataframe
    """
    l_values_total = []
    l_freqs_total = []
    for row in sensor_col:
        l_freqs, l_values = compute_welch_periodogram(
            values=row,
            window_type=window_type,
            sampling_frequency=sampling_frequency,
            segment_length_s = segment_length_s,
            overlap = overlap,
            spectral_resolution = spectral_resolution)
        l_values_total.append(l_values)
        l_freqs_total.append(l_freqs)

    return l_freqs_total, l_values_total

def compute_spectrogram(
        values: list,
        window_type = 'hann',
        sampling_frequency: int = 100,
        segment_length_s = 2,
        overlap: int = 0.8,
    )-> tuple:
    
    """Compute the spectrogram of the gyroscope signal
    
    Parameters
    ----------
    values: List
        The sensor column to be transformed (e.g. x-axis of gyroscope)
    window_type: str
        The type of window to be used for the spectrogram (default: 'hann')
    sampling_frequency: int
        The sampling frequency of the signal (default: 100)
    segment_length_s: int
        The length of each segment in seconds
    overlap: int
        The overlap between segments (fraction)
        
    Returns
    -------
    tuple
        Lists of spectrogram values
    """

    segment_length_n = sampling_frequency*segment_length_s
    overlap_n = segment_length_n*overlap
    window = signal.get_window(window_type,segment_length_n)

    f, t, S1 = signal.stft(values, fs=sampling_frequency, window=window, nperseg=segment_length_n, noverlap=overlap_n,boundary=None)
    S = np.abs(S1)*sampling_frequency 
    return S

def signal_to_spectrogram(
        sensor_col: pd.Series,
        window_type = 'hann',
        sampling_frequency: int = 100,
        segment_length_s = 2,
        overlap: int = 0.8,
    ) -> tuple:
    """Spectrogram of a signal per window.

    Parameters
    ----------
    sensor_col: pd.Series
        The sensor column to be transformed (e.g. x-axis of gyroscope)
    window_type: str
        The type of window to be used for the spectrogram (default: 'hann')
    sampling_frequency: int
        The sampling frequency of the signal (default: 100)
    segment_length_s: int
        The length of each segment in seconds
    overlap: int
        The overlap between segments (fraction)
    
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
            overlap = overlap
            )
        spectrogram.append(spectrogram_values)

    return spectrogram

def melscale(x):
    y = 64.875 * np.log10(1 + x/17.5)
    return y

def inverse_melscale(x):
    y = 17.5 * (10**(x/64.875) - 1)
    return y
    
def generate_mel_frequency_cepstral_coefficients(
        spectrogram: pd.Series,
        segment_length_s: 2,
        sampling_frequency: int = 100,
        low_frequency: int = 0,
        high_frequency: int = 25,
        n_filters: int = 15,
        n_coefficients: int = 12,
        ) -> pd.DataFrame:
    """Generate mel-frequency cepstral coefficients from the total power of the signal.
    
    Parameters
    ----------
    spectrogram: pd.Series
        The total spectrogram of the signal, extracted using stft
    window_length_s: int
        The number of seconds a segment constitutes
    sampling_frequency: int
        The sampling frequency of the data (default: 100)
    low_frequency: int
        The lower bound of the frequency band (default: 0)
    high_frequency: int
        The upper bound of the frequency band (default: 25)
    n_filters: int
        The number of DCT filters (default: 15)
    n_coefficients: int
        The number of coefficients to extract (default: 12)
    
    Returns
    -------
    pd.DataFrame
        A dataframe with a single column corresponding to a single mel-frequency cepstral coefficient
    """
    cepstral_coefs_total = []
    
    # construct filterbank
    segment_length = segment_length_s * sampling_frequency 
    freqs = np.linspace(melscale(low_frequency), melscale(high_frequency), num=n_filters+2) # equal intervals in mel scale
    freqs_melscale = inverse_melscale(freqs) # convert to frequency domain
    filter_points = np.round((segment_length / sampling_frequency * freqs_melscale)).astype(int) + 1 # rounding the filter edges

    filters = np.zeros((len(filter_points)-2, int(segment_length/2+1)))
    for j in range(len(filter_points)-2):
        filters[j, filter_points[j] : filter_points[j+2]] = signal.windows.triang(filter_points[j+2] - filter_points[j]) # triangular filters based on edges
        filters[j,:] /= (sampling_frequency/segment_length * np.sum(filters[j,:])) # normalization of the filter coefficients
    
    # construct dct filter
    dct_filters = np.empty((n_coefficients, n_filters))
    dct_filters[0, :] = 1.0 / np.sqrt(n_filters)
    samples = np.arange(1, 2 * n_filters, 2) * np.pi / (2.0 * n_filters)
    for i in range(1, n_coefficients):
        dct_filters[i, :] = np.cos(i * samples) * np.sqrt(2.0 / n_filters)

    for spectrogram_window in spectrogram:
        # filter signal
        power_filtered = np.array([np.dot(filters, spectrogram_window[:, i]) for i in range(spectrogram_window.shape[1])])
        log_power_filtered = np.log10(power_filtered)

        # generate cepstral coefficients
        cepstral_coefs = np.array([np.dot(dct_filters, x) for x in log_power_filtered])
        cepstral_coefs_total.append(np.transpose(np.mean(cepstral_coefs,axis=0)))

    return pd.DataFrame(cepstral_coefs_total, columns=['mfcc_{}'.format(j+1) for j in range(n_coefficients)])

def extract_frequency_peak(
    total_psd: pd.Series,
    freq_vect: pd.Series,
    min_frequency: int = 1,
    max_frequency: int = 25
    ):

    frequency_peak = []
    
    freq_range = freq_vect[0]
    freq_idx = np.where((freq_range>=min_frequency) & (freq_range<=max_frequency))
    for psd_window in total_psd: 
        peak_idx = np.argmax(psd_window[freq_idx])
        freq_peak = freq_range[peak_idx]+min_frequency
        frequency_peak.append(freq_peak)

    return frequency_peak

def extract_low_freq_power(
    total_psd: pd.Series,
    freq_vect: pd.Series,
    min_frequency: int = 0.5,
    max_frequency: int = 3,
    spectral_resolution: int = 0.25
    ):

    low_freq_power = []
    
    freq_range = freq_vect[0]
    freq_idx = np.where((freq_range>=min_frequency) & (freq_range<max_frequency))
    for psd_window in total_psd: 
        bandpower = spectral_resolution*np.sum(psd_window[freq_idx])
        low_freq_power.append(bandpower)

    return low_freq_power

def extract_tremor_power(
    total_psd: pd.Series,
    freq_vect: pd.Series,
    min_frequency: int = 3,
    max_frequency: int = 7,
    spectral_resolution: int = 0.25
    ):

    tremor_power = []
    
    freq_range = freq_vect[0]
    freq_idx = np.where((freq_range>=min_frequency) & (freq_range<=max_frequency))
    for psd_window in total_psd: 
        peak_idx = np.argmax(psd_window[freq_idx]) + np.min(freq_idx)
        left_idx =  np.max([0,int(peak_idx - 0.5/spectral_resolution)])
        right_idx = int(peak_idx + 0.5/spectral_resolution)
        peak_power = spectral_resolution*np.sum(psd_window[left_idx:right_idx+1])
        tremor_power.append(peak_power)

    return tremor_power

def extract_spectral_domain_features(config, df_windowed):

    # transform the temporal signal to the spectral domain using Welch's method
    for col in config.l_gyroscope_cols:
        df_windowed[f'{col}_freqs_PSD'], df_windowed[f'{col}_PSD'] = signal_to_PSD(
            sensor_col = df_windowed[col], 
            sampling_frequency = config.sampling_frequency,
            window_type = config.window_type, 
            segment_length_s = config.segment_length_s_psd,
            overlap = config.overlap,
            spectral_resolution = config.spectral_resolution_psd
            )
        df_windowed[f'{col}_spectrogram'] = signal_to_spectrogram(
            sensor_col = df_windowed[col], 
            sampling_frequency = config.sampling_frequency,
            window_type = config.window_type, 
            segment_length_s = config.segment_length_s_mfcc,
            overlap = config.overlap
            )
    
    df_windowed['total_PSD'] = df_windowed.apply(lambda x: sum(x[y+'_PSD'] for y in config.l_gyroscope_cols), axis=1) # sum PSD over the axes   
    df_windowed['total_spectrogram'] = df_windowed.apply(lambda x: sum(x[y+'_spectrogram'] for y in config.l_gyroscope_cols), axis=1) # sum spectrogram over the axes

    # compute the cepstral coefficients of the total power signal
    mfcc_cols = generate_mel_frequency_cepstral_coefficients(
        spectrogram=df_windowed['total_spectrogram'],
        segment_length_s = config.segment_length_s_mfcc,
        sampling_frequency=config.sampling_frequency,
        low_frequency=config.mfcc_low_frequency,
        high_frequency=config.mfcc_high_frequency,
        n_filters=config.n_dct_filters_mfcc,
        n_coefficients=config.n_coefficients_mfcc
        )
    
    df_windowed = pd.concat([df_windowed, mfcc_cols], axis=1)

    # compute the frequency of the peak in the PSD
    df_windowed['freq_peak'] = extract_frequency_peak(
        total_psd = df_windowed['total_PSD'],
        freq_vect = df_windowed['gyroscope_x_freqs_PSD'],
        min_frequency = config.peak_min_frequency,
        max_frequency = config.peak_max_frequency
    )
    
    df_windowed['low_freq_power'] = extract_low_freq_power(
        total_psd = df_windowed['total_PSD'],
        freq_vect = df_windowed['gyroscope_x_freqs_PSD'],
        min_frequency = config.low_power_min_frequency,
        max_frequency = config.low_power_max_frequency,
        spectral_resolution = config.spectral_resolution_psd
    )

    df_windowed['tremor_power'] = extract_tremor_power(
        total_psd = df_windowed['total_PSD'],
        freq_vect = df_windowed['gyroscope_x_freqs_PSD'],
        min_frequency = config.tremor_power_min_frequency,
        max_frequency = config.tremor_power_max_frequency,
        spectral_resolution = config.spectral_resolution_psd 
    )

    df_windowed = df_windowed.rename(columns={'window_start': DataColumns.TIME})

    
    return df_windowed
    