import tsdf
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union
from scipy.stats import gaussian_kde

from paradigma.classification import ClassifierPackage
from paradigma.constants import DataColumns
from paradigma.config import TremorConfig
from paradigma.tremor.feature_extraction import extract_spectral_domain_features
from paradigma.segmenting import tabulate_windows, WindowedDataExtractor
from paradigma.util import get_end_iso8601, write_df_data, read_metadata, aggregate_parameter


def extract_tremor_features(df: pd.DataFrame, config: TremorConfig) -> pd.DataFrame:
    """
    This function groups sequences of timestamps into windows and subsequently extracts 
    tremor features from windowed gyroscope data.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing sensor data, which includes time and gyroscope data. The data should be
        structured with the necessary columns as specified in the `config`.

    config : TremorConfig
        Configuration object containing parameters for feature extraction, including column names for time, gyroscope data,
        as well as settings for windowing, and feature computation.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing extracted tremor features and a column corresponding to time.
    
    Notes
    -----
    - This function groups the data into windows based on timestamps.
    - The input DataFrame must include columns as specified in the `config` object for proper feature extraction.

    Raises
    ------
    ValueError
        If the input DataFrame does not contain the required columns as specified in the configuration or if any step in the feature extraction fails.
    """
    # group sequences of timestamps into windows
    windowed_cols = [DataColumns.TIME] + config.gyroscope_cols
    windowed_data = tabulate_windows(df, windowed_cols, config.window_length_s, config.window_step_length_s, config.sampling_frequency)

    extractor = WindowedDataExtractor(windowed_cols)

    # Extract the start time and gyroscope data from the windowed data
    idx_time = extractor.get_index(DataColumns.TIME)
    idx_gyro = extractor.get_slice(config.gyroscope_cols)

    # Extract data
    start_time = np.min(windowed_data[:, :, idx_time], axis=1)
    windowed_gyro = windowed_data[:, :, idx_gyro]

    df_features = pd.DataFrame(start_time, columns=[DataColumns.TIME])
    
    # transform the signals from the temporal domain to the spectral domain and extract tremor features
    df_spectral_features = extract_spectral_domain_features(config, windowed_gyro)

    # Combine spectral features with the start time
    df_features= pd.concat([df_features, df_spectral_features], axis=1)

    return df_features

def extract_tremor_features_io(input_path: Union[str, Path], output_path: Union[str, Path], config: TremorConfig) -> None:
    # Load data
    metadata_time, metadata_values = read_metadata(input_path, config.meta_filename, config.time_filename, config.values_filename)
    df = tsdf.load_dataframe_from_binaries([metadata_time, metadata_values], tsdf.constants.ConcatenationType.columns)

    # Extract tremor features
    df_windowed = extract_tremor_features(df, config)

    # Store data
    end_iso8601 = get_end_iso8601(start_iso8601=metadata_time.start_iso8601,
                                  window_length_seconds=int(df_windowed[DataColumns.TIME][-1:].values[0] + config.window_length_s))

    metadata_values.end_iso8601 = end_iso8601
    metadata_values.file_name = 'tremor_values.bin'
    metadata_time.end_iso8601 = end_iso8601
    metadata_time.file_name = 'tremor_time.bin'

    metadata_values.channels = list(config.d_channels_values.keys())
    metadata_values.units = list(config.d_channels_values.values())

    metadata_time.channels = [DataColumns.TIME]
    metadata_time.units = ['relative_time_ms']

    write_df_data(metadata_time, metadata_values, output_path, 'tremor_meta.json', df_windowed)


def detect_tremor(df: pd.DataFrame, config: TremorConfig, full_path_to_classifier_package: Union[str, Path]) -> pd.DataFrame:
    """
    Detects tremor in the input DataFrame using a pre-trained classifier and applies a threshold to the predicted probabilities.

    This function performs the following steps:
    1. Loads the pre-trained classifier and scaling parameters from the provided directory.
    2. Scales the relevant features in the input DataFrame (`df`) using the loaded scaling parameters.
    3. Makes predictions using the classifier to estimate the probability of tremor.
    4. Applies a threshold to the predicted probabilities to classify whether tremor is detected or not.
    5. Checks for rest tremor by verifying the frequency of the peak and low-frequency power.
    6. Adds the predicted probabilities and the classification result to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing extracted tremor features. The DataFrame must include
        the necessary columns as specified in the classifier's feature names.

    config : TremorConfig
        Configuration object containing settings for tremor detection, including the frequency range for rest tremor.

    full_path_to_classifier_package : Union[str, Path]
        The path to the directory containing the classifier file, threshold value, scaler parameters, and other necessary input
        files for tremor detection.

    Returns
    -------
    pd.DataFrame
        The input DataFrame (`df`) with two additional columns:
        - `PRED_TREMOR_PROBA`: Predicted probability of tremor based on the classifier.
        - `PRED_TREMOR_LOGREG`: Binary classification result (True for tremor, False for no tremor), based on the threshold applied to `PRED_TREMOR_PROBA`.
        - `PRED_TREMOR_CHECKED`: Binary classification result (True for tremor, False for no tremor), after performing extra checks for rest tremor on `PRED_TREMOR_LOGREG`.
        - `PRED_ARM_AT_REST`: Binary classification result (True for arm at rest or stable posture, False for significant arm movement), based on the low-frequency power.

    Notes
    -----
    - The threshold used to classify tremor is loaded from a file and applied to the predicted probabilities.

    Raises
    ------
    FileNotFoundError
        If the classifier, scaler, or threshold files are not found at the specified paths.
    ValueError
        If the DataFrame does not contain the expected features for prediction or if the prediction fails.

    """

    # Load the classifier package
    clf_package = ClassifierPackage.load(full_path_to_classifier_package)

    # Set classifier
    clf = clf_package.classifier
    feature_names_scaling = clf_package.scaler.feature_names_in_
    feature_names_predictions = clf.feature_names_in_

    # Apply scaling to relevant columns
    scaled_features = clf_package.transform_features(df.loc[:, feature_names_scaling])

    # Replace scaled features in a copy of the relevant features for prediction
    X = df.loc[:, feature_names_predictions].copy()
    X.loc[:, feature_names_scaling] = scaled_features

    # Get the tremor probability 
    df[DataColumns.PRED_TREMOR_PROBA] = clf_package.predict_proba(X)

    # Make prediction based on pre-defined threshold
    df[DataColumns.PRED_TREMOR_LOGREG] = (df[DataColumns.PRED_TREMOR_PROBA] >= clf_package.threshold).astype(int)

    # Perform extra checks for rest tremor 
    peak_check = (df['freq_peak'] >= config.fmin_tremor_power) & (df['freq_peak']<=config.fmax_tremor_power) # peak within 3-7 Hz
    df[DataColumns.PRED_ARM_AT_REST] = (df['low_freq_power'] <= config.movement_threshold).astype(int) # arm at rest or in stable posture
    df[DataColumns.PRED_TREMOR_CHECKED] = ((df[DataColumns.PRED_TREMOR_LOGREG]==1) & (peak_check==True) & (df[DataColumns.PRED_ARM_AT_REST] == True)).astype(int)
    
    return df


def detect_tremor_io(input_path: Union[str, Path], output_path: Union[str, Path], path_to_classifier_input: Union[str, Path], config: TremorConfig) -> None:
    
    # Load the data
    config.set_filenames('tremor')

    metadata_time, metadata_values = read_metadata(input_path, config.meta_filename, config.time_filename, config.values_filename)
    df = tsdf.load_dataframe_from_binaries([metadata_time, metadata_values], tsdf.constants.ConcatenationType.columns)

    df = detect_tremor(df, config, path_to_classifier_input)

    # Prepare the metadata
    metadata_values.file_name = 'tremor_values.bin'
    metadata_time.file_name = 'tremor_time.bin'

    metadata_values.channels = list(config.d_channels_values.keys())
    metadata_values.units = list(config.d_channels_values.values())

    metadata_time.channels = [DataColumns.TIME]
    metadata_time.units = ['relative_time_ms']

    write_df_data(metadata_time, metadata_values, output_path, 'tremor_meta.json', df)


def aggregate_tremor(df: pd.DataFrame, config: TremorConfig):
    """
    Quantifies the amount of tremor time and tremor power, aggregated over all windows in the input dataframe.
    Tremor time is calculated as the number of the detected tremor windows, as percentage of the number of windows 
    without significant non-tremor movement (at rest). For tremor power the following aggregates are derived:
    the median, mode and percentile of tremor power specified in the configuration object. 
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing extracted tremor features. The DataFrame must include
        the necessary columns as specified in the classifier's feature names.

    config : TremorConfig
        Configuration object containing the percentile for aggregating tremor power.

    Returns
    -------
    A json file with the aggregated tremor time and tremor power measures, as well as the total number of windows
    available in the input dataframe, and the number of windows at rest.

    Notes
    -----
    - Tremor power is converted to log scale, after adding a constant of 1, so that zero tremor power
    corresponds to a value of 0 in log scale.
    - The modal tremor power is computed based on gaussian kernel density estimation.
  
    """

    nr_windows_total = df.shape[0] # number of windows in the input dataframe

    # remove windows with detected non-tremor arm movements to control for the amount of arm activities performed
    df_filtered = df.loc[df.pred_arm_at_rest == 1]
    nr_windows_rest = df_filtered.shape[0] # number of windows without non-tremor arm movement

    # calculate tremor time
    perc_windows_tremor= np.sum(df_filtered['pred_tremor_checked']) / nr_windows_rest * 100 # as percentage of total measured time without non-tremor arm movement

    # calculate aggregated tremor power measures
    tremor_power = df_filtered.loc[df_filtered['pred_tremor_checked'] == 1, 'tremor_power']
    tremor_power = np.log10(tremor_power+1) # convert to log scale
    aggregated_tremor_power = {}
    
    for aggregate in config.aggregates_tremor_power:
        aggregate_name = f"{aggregate}_tremor_power"
        if aggregate == 'mode':
            # calculate modal tremor power
            bin_edges = np.linspace(0, 6, 301)
            kde = gaussian_kde(tremor_power)
            kde_values = kde(bin_edges)
            max_index = np.argmax(kde_values)
            aggregated_tremor_power['modal_tremor_power'] = bin_edges[max_index]
        else: # calculate te other aggregates (e.g. median and 90th percentile) of tremor power
            aggregated_tremor_power[aggregate_name] = aggregate_parameter(tremor_power, aggregate)
    
    # store aggregates in json format
    d_aggregates = {
        'metadata': {
            'nr_windows_total': nr_windows_total,
            'nr_windows_rest': nr_windows_rest
        },
        'aggregated_tremor_measures': {
            'perc_windows_tremor': perc_windows_tremor,
            'median_tremor_power': aggregated_tremor_power['median_tremor_power'],
            'modal_tremor_power': aggregated_tremor_power['modal_tremor_power'],
            '90p_tremor_power': aggregated_tremor_power['90p_tremor_power']
        }
    }

    return d_aggregates


def aggregate_tremor_io(path_to_feature_input: Union[str, Path], path_to_prediction_input: Union[str, Path], output_path: Union[str, Path], config: TremorConfig) -> None:
    
    # Load the features & predictions
    metadata_time, metadata_values = read_metadata(path_to_feature_input, config.meta_filename, config.time_filename, config.values_filename)
    df_features = tsdf.load_dataframe_from_binaries([metadata_time, metadata_values], tsdf.constants.ConcatenationType.columns)

    metadata_dict = tsdf.load_metadata_from_path(path_to_prediction_input / config.meta_filename)
    metadata_time = metadata_dict[config.time_filename]
    metadata_values = metadata_dict[config.values_filename]
    df_predictions = tsdf.load_dataframe_from_binaries([metadata_time, metadata_values], tsdf.constants.ConcatenationType.columns)

    # Subset features
    df_features = df_features[['tremor_power', 'low_freq_power']]

    # Concatenate predictions and tremor power
    df = pd.concat([df_predictions, df_features], axis=1)

    # Compute aggregated tremor measures
    d_aggregates = aggregate_tremor(df, config)

    # Save output
    with open(output_path / "tremor_aggregates.json", 'w') as json_file:
        json.dump(d_aggregates, json_file, indent=4)
