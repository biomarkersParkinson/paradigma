import numpy as np
import pandas as pd

from scipy import signal, fft

def compute_fft(
        values: list,
        window_type: str = 'hann',
        sampling_frequency: int = 100,
    ) -> tuple:
    """Compute the Fast Fourier Transform (FFT) of a signal.
    
    Parameters
    ----------
    values: list
        The values of the signal (e.g., accelerometer data) of a single window.
    window_type: str
        The type of window to be used for the FFT (default: 'hann')
    sampling_frequency: int
        The sampling frequency of the signal (default: 100)
        
    Returns
    -------
    tuple
        The FFT values and the corresponding frequencies
    """
    w = signal.get_window(window_type, len(values), fftbins=False)
    yf = 2*fft.fft(values*w)[:int(len(values)/2+1)]
    xf = fft.fftfreq(len(values), 1/sampling_frequency)[:int(len(values)/2+1)]

    return yf, xf

def signal_to_ffts(
        sensor_col: pd.Series,
        window_type: str = 'hann',
        sampling_frequency: int = 100,
    ) -> tuple:
    """Compute the Fast Fourier Transform (FFT) of a signal per window (can probably be combined with compute_fft and simplified).

    Parameters
    ----------
    sensor_col: pd.Series
        The sensor column to be transformed (e.g. x-axis of gyroscope)
    window_type: str
        The type of window to be used for the FFT (default: 'hann')
    sampling_frequency: int
        The sampling frequency of the signal (default: 100)
    
    Returns
    -------
    tuple
        Lists of FFT values and corresponding frequencies which can be concatenated as column to the dataframe
    """
    l_values_total = []
    l_freqs_total = []
    for row in sensor_col:
        l_values, l_freqs = compute_fft(
            values=row,
            window_type=window_type,
            sampling_frequency=sampling_frequency)
        l_values_total.append(l_values)
        l_freqs_total.append(l_freqs)

    return l_freqs_total, l_values_total
    
def compute_power(
        df: pd.DataFrame,
        fft_cols: list
    ) -> pd.Series:
    """Compute the power of the FFT values.
    
    Parameters
    ----------
    df: pd.DataFrame
        The dataframe containing the FFT values
    fft_cols: list
        The names of the columns containing the FFT values
    
    Returns
    -------
    pd.Series
        The power of the FFT values
    """
    for col in fft_cols:
        df['{}_power'.format(col)] = df[col].apply(lambda x: np.square(np.abs(x)))

    return df.apply(lambda x: sum([np.array([y for y in x[col+'_power']]) for col in fft_cols]), axis=1)


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
        The total power of the signal, extracted using compute_power
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


def extract_spectral_domain_features(config, df_windowed,l_sensor_colnames):

    for col in l_sensor_colnames:

        # transform the temporal signal to the spectral domain using the fast fourier transform
        df_windowed[f'{col}_freqs'], df_windowed[f'{col}_fft'] = signal_to_ffts(
            sensor_col=df_windowed[col],
            window_type=config.window_type,
            sampling_frequency=config.sampling_frequency
            )
    
    # compute the power summed over the individual axes
    df_windowed['total_power'] = compute_power(
        df=df_windowed,
        fft_cols=[f'{col}_fft' for col in l_sensor_colnames])

    # compute the cepstral coefficients of the total power signal
    mfcc_cols = generate_mel_frequency_cepstral_coefficients(
        total_power_col=df_windowed['total_power'],
        window_length_s=config.window_length_s,
        sampling_frequency=config.sampling_frequency,
        low_frequency=config.spectrum_low_frequency,
        high_frequency=config.spectrum_high_frequency,
        n_filters=config.n_dct_filters_cc,
        n_coefficients=config.n_coefficients_cc
        )
    
    df_windowed = pd.concat([df_windowed, mfcc_cols], axis=1)
    
    return df_windowed
    