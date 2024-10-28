from typing import List
import numpy as np
import pandas as pd

from scipy import signal

from paradigma.constants import DataColumns

def compute_welch_periodogram(
        values: list,
        sampling_frequency: int = 100,
        window_type = 'hann',
        segment_length_s = 2,
        overlap_s: int = 0.5,
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
    overlap_s: int
        The overlap between segments in seconds
    spectral_resolution: int
        The frequency resolution in Hz
        
    Returns
    -------
    tuple
        Lists of PSD values and the corresponding frequencies
    """

    segment_length_n = sampling_frequency*segment_length_s
    overlap_n = sampling_frequency*overlap_s
    window = signal.get_window(window_type, segment_length_n, fftbins=False)
    nfft = sampling_frequency/spectral_resolution
    
    f, Pxx = signal.welch(values,sampling_frequency,window,segment_length_n,
                          overlap_n,nfft,detrend=False)

    return f, Pxx

def signal_to_PSD(
        sensor_col: pd.Series,
        sampling_frequency: int = 100,
        window_type = 'hann',
        segment_length_s = 2,
        overlap_s: int = 0.5,
        spectral_resolution = 0.25
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
    overlap_s: int
        The overlap between segments in seconds
    
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
            overlap_s = overlap_s,
            spectral_resolution = spectral_resolution)
        l_values_total.append(l_values)
        l_freqs_total.append(l_freqs)

    return l_freqs_total, l_values_total

def melscale(x):
    y = 64.875 * np.log10(1 + x/17.5)
    return y

def inverse_melscale(x):
    y = 17.5 * (10**(x/64.875) - 1)
    return y
    

def generate_mel_frequency_cepstral_coefficients(
        total_power_col: pd.Series,
        window_length_s: int,
        sampling_frequency: int = 100,
        low_frequency: int = 0,
        high_frequency: int = 25,
        n_filters: int = 20,
        n_coefficients: int = 12,
        ) -> pd.DataFrame:
    """Generate mel-frequency cepstral coefficients from the total power of the signal.
    
    Parameters
    ----------
    total_power_col: pd.Series
        The total power of the signal, extracted using fourier
    window_length_s: int
        The number of seconds a window constitutes
    sampling_frequency: int
        The sampling frequency of the data (default: 100)
    low_frequency: int
        The lower bound of the frequency band (default: 0)
    high_frequency: int
        The upper bound of the frequency band (default: 25)
    n_filters: int
        The number of DCT filters (default: 20)
    n_coefficients: int
        The number of coefficients to extract (default: 12)
    
    Returns
    -------
    pd.DataFrame
        A dataframe with a single column corresponding to a single mel-frequency cepstral coefficient
    """
    window_length = window_length_s * sampling_frequency
    
    # compute filter points
    freqs = np.linspace(melscale(low_frequency), melscale(high_frequency), num=n_filters+2)
    freqs_melscale = inverse_melscale(freqs)
    filter_points = np.floor((window_length + 1) / sampling_frequency * freqs_melscale).astype(int)  

    # construct filterbank
    filters = np.zeros((len(filter_points)-2, int(window_length/2+1)))
    for j in range(len(filter_points)-2):
        filters[j, filter_points[j] : filter_points[j+1]] = np.linspace(0, 1, filter_points[j+1] - filter_points[j])
        filters[j, filter_points[j+1] : filter_points[j+2]] = np.linspace(1, 0, filter_points[j+2] - filter_points[j+1])

    # filter signal
    power_filtered = [np.dot(filters, x) for x in total_power_col]
    log_power_filtered = [10.0 * np.log10(x) for x in power_filtered]

    # generate cepstral coefficients
    dct_filters = np.empty((n_coefficients, n_filters))
    dct_filters[0, :] = 1.0 / np.sqrt(n_filters)

    samples = np.arange(1, 2 * n_filters, 2) * np.pi / (2.0 * n_filters)

    for i in range(1, n_coefficients):
        dct_filters[i, :] = np.cos(i * samples) * np.sqrt(2.0 / n_filters)

    cepstral_coefs = [np.dot(dct_filters, x) for x in log_power_filtered]

    return pd.DataFrame(np.vstack(cepstral_coefs), columns=['mfcc_{}'.format(j+1) for j in range(n_coefficients)])


def extract_spectral_domain_features(config, df_windowed):

    # transform the temporal signal to the spectral domain using Welch's method
    for col in config.l_gyroscope_cols:
        df_windowed[f'{col}_freqs'], df_windowed[f'{col}_PSD'] = signal_to_PSD(
            sensor_col = df_windowed[col], 
            sampling_frequency = config.sampling_frequency,
            window_type = config.window_type, 
            segment_length_s = config.segment_length_s,
            overlap_s = config.overlap_s
            )
    
    df_windowed['total_PSD'] = df_windowed.apply(lambda x: sum(x[y+'_PSD'] for y in config.l_gyroscope_cols), axis=1)

    print(df_windowed['gyroscope_x_freqs'][0])

    # compute the cepstral coefficients of the total power signal
    mfcc_cols = generate_mel_frequency_cepstral_coefficients(
        total_power_col=df_windowed['total_PSD'],
        window_length_s=config.window_length_s,
        sampling_frequency=config.sampling_frequency,
        low_frequency=config.spectrum_low_frequency,
        high_frequency=config.spectrum_high_frequency,
        n_filters=config.n_dct_filters_cc,
        n_coefficients=config.n_coefficients_cc
        )
    
    df_windowed = pd.concat([df_windowed, mfcc_cols], axis=1)

    df_windowed = df_windowed.rename(columns={'window_start': DataColumns.TIME})

    
    return df_windowed
    