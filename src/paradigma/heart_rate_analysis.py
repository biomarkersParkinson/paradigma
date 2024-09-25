import numpy as np
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from dateutil import parser

import tsdf
import tsdf.constants
from paradigma.heart_rate_analysis_config import HeartRateFeatureExtractionConfig
from paradigma.heart_rate_util import extract_ppg_features, calculate_power_ratio, read_PPG_quality_classifier
from paradigma.util import read_metadata, write_np_data
from paradigma.constants import DataColumns, UNIX_TICKS_MS, DataUnits, TimeUnit


def extract_signal_quality_features(input_path: str, classifier_path: str, output_path: str, config: HeartRateFeatureExtractionConfig) -> None:
    # load data
    metadata_time_ppg, metadata_samples_ppg = read_metadata(input_path, "PPG_meta.json", "PPG_time.bin", "PPG_samples.bin")
    df_ppg = tsdf.load_dataframe_from_binaries([metadata_time_ppg, metadata_samples_ppg], tsdf.constants.ConcatenationType.columns)
    arr_ppg = df_ppg[DataColumns.PPG].to_numpy()
    relative_time_ppg = df_ppg[DataColumns.TIME].to_numpy()
    
    metadata_time_acc, metadata_samples_acc = read_metadata(input_path, "accelerometer_meta.json", "accelerometer_time.bin", "accelerometer_samples.bin")
    df_acc = tsdf.load_dataframe_from_binaries([metadata_time_acc, metadata_samples_acc], tsdf.constants.ConcatenationType.columns)
    arr_acc = df_acc[[DataColumns.ACCELEROMETER_X, DataColumns.ACCELEROMETER_Y, DataColumns.ACCELEROMETER_Z]].to_numpy()

    sampling_frequency_ppg = config.sampling_frequency_ppg
    sampling_frequency_imu = config.sampling_frequency_imu

    # Parameters
    epoch_length = 6  # in seconds
    overlap = 5  # in seconds

    # Number of samples in epoch
    samples_per_epoch_ppg = int(epoch_length * sampling_frequency_ppg)
    samples_per_epoch_acc = int(epoch_length * sampling_frequency_imu)

    # Calculate number of samples to shift for each epoch
    samples_shift_ppg = int((epoch_length - overlap) * sampling_frequency_ppg)
    samples_shift_acc = int((epoch_length - overlap) * sampling_frequency_imu)

    pwelchwin_acc = int(3 * sampling_frequency_imu)  # window length for pwelch
    pwelchwin_ppg = int(3 * sampling_frequency_ppg)  # window length for pwelch
    noverlap_acc = int(0.5 * pwelchwin_acc)  # overlap for pwelch
    noverlap_ppg = int(0.5 * pwelchwin_ppg)  # overlap for pwelch

    f_bin_res = 0.05  # the threshold is set based on this binning
    nfft_ppg = np.arange(0, sampling_frequency_ppg / 2, f_bin_res)  # frequency bins for pwelch ppg
    nfft_acc = np.arange(0, sampling_frequency_imu / 2, f_bin_res)  # frequency bins for pwelch imu

    features_ppg_scaled = []
    feature_acc = []
    t_unix_feat_total = []
    count = 0
    acc_idx = 0

    # Read the classifier (it contains mu and sigma)
    clf = read_PPG_quality_classifier(classifier_path)
    # not used here: lr_model = clf['model']
    arr_mu = clf['mu'][:, 0]
    arr_sigma = clf['sigma'][:, 0]

    scaler = StandardScaler()
    scaler.mean_ = arr_mu
    scaler.scale_ = arr_sigma

    ppg_start_time = parser.parse(metadata_time_ppg.start_iso8601)

    # Loop over 6s segments for both PPG and IMU and calculate features
    for i in range(0, len(arr_ppg) - samples_per_epoch_ppg + 1, samples_shift_ppg):
        if acc_idx + samples_per_epoch_acc > len(arr_acc):  # For the last epoch, check if the segment for IMU is too short (not 6 seconds)
            break
        else:
            acc_segment = arr_acc[acc_idx:acc_idx + samples_per_epoch_acc, :]  # Extract the IMU window (6 seconds)

        ppg_segment = arr_ppg[i:i + samples_per_epoch_ppg]  # Extract the PPG window (6 seconds)

        count += 1

        # Feature extraction + scaling
        features = extract_ppg_features(ppg_segment, sampling_frequency_ppg)
        features = features.reshape(1, -1)
        features_ppg_scaled.append(scaler.transform(features)[0])

        # Calculating PSD (power spectral density) of IMU and PPG
        f1, pxx1 = welch(acc_segment, sampling_frequency_imu, window='hann', nperseg=pwelchwin_acc, noverlap=None, nfft=len(nfft_acc))
        PSD_imu = np.sum(pxx1, axis=0)  # sum over the three axes
        f2, pxx2 = welch(ppg_segment, sampling_frequency_ppg, window='hann', nperseg=pwelchwin_ppg, noverlap=None, nfft=len(nfft_ppg))
        PSD_ppg = np.sum(pxx2)  # this does nothing, equal to PSD_ppg = pxx2

        # IMU feature extraction
        feature_acc.append(calculate_power_ratio(f1, PSD_imu, f2, PSD_ppg))  # Calculate the power ratio of the accelerometer signal in the PPG frequency range

        # time channel
        t_unix_feat_total.append((relative_time_ppg[i] + ppg_start_time.timestamp()) * UNIX_TICKS_MS)  # Save in absolute unix time ms
        acc_idx += samples_shift_acc  # update IMU_idx

    # Convert lists to numpy arrays
    features_ppg_scaled = np.array(features_ppg_scaled)
    feature_acc = np.array(feature_acc)
    t_unix_feat_total = np.array(t_unix_feat_total)

    # Synchronization information
    # TODO: store this, as this is needed for the HR pipeline
    # v_sync_ppg_total = np.array([
    #     ppg_indices[0],  # start index
    #     ppg_indices[1],  # end index
    #     segment_ppg[0],  # Segment index
    #     count  # Number of epochs in the segment
    # ])

    metadata_features_ppg = metadata_samples_ppg
    metadata_features_ppg.channels = [
    DataColumns.VARIANCE,
    DataColumns.MEAN,
    DataColumns.MEDIAN,
    DataColumns.KURTOSIS,
    DataColumns.SKEWNESS,
    DataColumns.DOMINANT_FREQUENCY,
    DataColumns.RELATIVE_POWER,
    DataColumns.SPECTRAL_ENTROPY,
    DataColumns.SIGNAL_NOISE_RATIO,
    DataColumns.SECOND_HIGHEST_PEAK
]
    metadata_features_ppg.units = [
    DataUnits.NONE,  # variance
    DataUnits.NONE,  # mean
    DataUnits.NONE,  # median
    DataUnits.NONE,  # kurtosis
    DataUnits.NONE,  # skewness
    DataUnits.FREQUENCY,  # dominant_frequency (measured in Hz)
    DataUnits.NONE,  # relative_power (unitless measure)
    DataUnits.NONE,  # spectral_entropy (unitless measure)
    DataUnits.NONE,  # signal_noise_ratio (unitless measure)
    DataUnits.NONE   # second_highest_peak (depends on the feature, assume no unit)
]
    metadata_time_ppg.channels = [DataColumns.TIME]
    metadata_time_ppg.units = [TimeUnit.RELATIVE_MS]
    
    metadata_features_acc = metadata_samples_acc
    metadata_features_acc.channels = [DataColumns.POWER_RATIO]
    metadata_features_acc.units = [DataUnits.NONE]

    metadata_time_acc.channels = [DataColumns.TIME]
    metadata_time_acc.units = [TimeUnit.RELATIVE_MS]

    write_np_data(metadata_time_ppg, t_unix_feat_total, metadata_features_ppg, 
                  features_ppg_scaled, output_path, 'features_ppg_meta.json')
    write_np_data(metadata_time_acc, t_unix_feat_total, metadata_features_acc,
                  feature_acc, output_path, 'feature_acc_meta.json')

