%% Main script to perform heart rate estimation of wearable PPG
% This script uses both PPG and accelerometer and performs the following
% steps:
%   1. Loading all metadata of PPG and IMU
%   2. Query on data availability + synchronization
%   3. Loading relevant segment sensor data using tsdf wrapper (start for loop over synchronized segment indices)
%   4. Synchronize the data (correct indices etc)
%   5. Data preprocessing
%   6. Perform pseude-smoothed wigner-ville distribution (PSWVD) on PPG
%   7. Saving the HR estimates in tsdf format

%% Initalization
% Setting data paths + extracting metafilenames already
clear all; close all; clc
addpath(genpath('..\..\..\dbpd-toolbox'))       % Add git repository to the path
addpath(genpath("..\..\..\\tsdf4matlab"))       % Add wrapper to the path
warning('off','all')        % Turn off warnings to improve speed in spwvd especially

% Setting the data paths
unix_ticks_ms = 1000.0;
fs_ppg = 30;     % Establish the sampling rate desired for resampling PPG --> now chosen to be fixed on 30 Hz

raw_data_root = '..\..\tests\data\1.sensor_data\';
ppp_data_path_ppg = [raw_data_root 'PPG\'];
meta_ppg = tsdf_scan_meta(ppp_data_path_ppg);            % tsdf_scan_meta returns metafile struct containing information of all metafiles from all patients in tsdf_dirlist
n_files_ppg = length(meta_ppg); 

sqa_data_path = '..\..\tests\data\4.predictions\ppg'; % Set the path to the SQA data
sqa_output_list = dir(fullfile(sqa_data_path, '*_meta.json'));              % seperate for the SQA output files

meta_path_sqa = fullfile(sqa_output_list.folder, sqa_output_list.name); % Get the name of the SQA output file
[metadata_list_sqa, data_list_sqa] = load_tsdf_metadata_from_path(meta_path_sqa); % Load the metadata of the SQA output file

sync_idx = tsdf_values_idx(metadata_list_sqa, 'sync'); % Get the index of the sync field in the SQA output file
data_sync = data_list_sqa{sync_idx};    % Get the sync data of the SQA output file
data_sync_zeros = all(data_sync == 0, 2); % Find rows containing only zeros --> in updated code this should not be possible anymore
data_sync(data_sync_zeros, :) = [];  % Remove rows containing only zeros

n_segments_sync = size(data_sync,1);  % Get the number of segments in the SQA output file

% Load the classification data
ppg_prob_idx = tsdf_values_idx(metadata_list_sqa, 'ppg'); % Get the index of the ppg field in the SQA output file
ppg_post_prob = data_list_sqa{ppg_prob_idx};    % Get the PPG probability data of the SQA output file

imu_idx = tsdf_values_idx(metadata_list_sqa, 'sqa_imu'); % Get the index of the imu field in the SQA output file
imu_label = data_list_sqa{imu_idx};    % Get the IMU label data of the SQA output file

% Calculate start_end indices of the classification corresponding to the
% correct segment --> needed since only number of segments is stored

% Initialize the start_end_indices array of the classification epochs
for i = 1:length(data_sync(:,4))
    if i == 1
        start_end_indices(i, 1) = 1;
    else
        start_end_indices(i, 1) = start_end_indices(i-1, 2) + 1;
    end
    start_end_indices(i, 2) = sum(data_sync(1:i,4));
end


%% Setting some parameters for HR analysis
min_window_length = 10;
min_hr_samples = min_window_length*fs_ppg;
threshold_sqa = 0.5;  
fs_ppg_est = [];        % Initialize the estimated sampling rate of the PPG

hr_est_length = 2;       %Estimation of HR is per 2 s (implemented in the PPG_TFD_HR function)
hr_est_samples = hr_est_length*fs_ppg;   % number of samples 

% Time-frequency distribution parameters
tfd_length = 10;  % Length of the epoch to calculate the time-frequency distribution in seconds
kern_type = 'sep';   % sep is the spwvd from the J. O'Toole box;)
win_type_doppler = 'hamm'; win_type_lag = 'hamm';
win_length_doppler = 1; win_length_lag = 8;
doppler_samples = fs_ppg * win_length_doppler; lag_samples = win_length_lag * fs_ppg;
kern_params = { {doppler_samples, win_type_doppler}, {lag_samples ,win_type_lag} };

% Initialze a moving average filter struct
MA = struct;
MA.value = 0;                                  % set MA to 1 if you want a moving average filter over the tfd from WVD and SPWVD to overcome unwanted fundamental frequency switching. 
MA.window = 30;                                % Window size for the filter (order = 30-1) --> is not really necessary to have in the struct but for simplicity it is
MA.FC = 1/MA.window*ones(MA.window,1);         % Set the filter coefficients for the MA filter

v_hr_ppg = [];
t_hr_unix = [];
%% Loop over all synchronized segments
for n = 1:n_segments_sync
    ppg_indices = data_sync(n,1:2); % Get the indices of the PPG segment
    ppg_segment = data_sync(n,3);   % Get the segment number of the PPG segment

    class_start = start_end_indices(n,1);
    class_end = start_end_indices(n,2); 

    meta_path_ppg = meta_ppg(ppg_segment).tsdf_meta_fullpath;

    [metadata_list_ppg, data_list_ppg] = load_tsdf_metadata_from_path(meta_path_ppg);

    time_idx_ppg = tsdf_values_idx(metadata_list_ppg, 'time');
    values_idx_ppg = tsdf_values_idx(metadata_list_ppg, 'samples');


    t_iso_ppg = metadata_list_ppg{time_idx_ppg}.start_iso8601;
    datetime_ppg = datetime(t_iso_ppg, 'InputFormat', 'dd-MMM-yyyy HH:mm:ss z', 'TimeZone', 'UTC');
    ts_ppg = posixtime(datetime_ppg) * unix_ticks_ms;

    t_ppg = cumsum(double(data_list_ppg{time_idx_ppg})) + ts_ppg;

    tr_ppg = (t_ppg - ts_ppg) / unix_ticks_ms;

    v_ppg = data_list_ppg{values_idx_ppg};

    clear t_ppg data_list_ppg

    v_ppg = v_ppg(ppg_indices(1):ppg_indices(2));

    tr_ppg = tr_ppg(ppg_indices(1):ppg_indices(2));

    ts_sync = ts_ppg + tr_ppg(1) * unix_ticks_ms;
    tr_ppg = tr_ppg - tr_ppg(1); 

    fs_ppg_est = 1/median(diff(tr_ppg)); 

    if length(v_ppg) < fs_ppg * min_window_length % Check if the sample is of sufficient length
        warning('Sample is of insufficient length!')
        continue
    else
        [v_ppg_pre, tr_ppg_pre] = preprocessing_ppg(tr_ppg, v_ppg, fs_ppg);
    end
    clear  v_ppg tr_ppg  % Clear the redundant variables to free up memory

    % Select the correct classification data
    class_ppg_segment = ppg_post_prob(class_start:class_end);
    class_acc_segment = imu_label(class_start:class_end);

    % Assign the window-level probabilities to the individual samples
    data_prob_sample = sample_prob_final(class_ppg_segment, class_acc_segment, fs_ppg); 

    sqa_label = [];

    for i = 1:length(data_prob_sample)
        if data_prob_sample(i) > threshold_sqa
            sqa_label(i,1) = 1;          % Assign label 1 --> high-quality
        else
            sqa_label(i,1) = 0;          % Assign low quality
        end
    end

    [v_start_idx, v_end_idx] = extract_hr_segments(sqa_label, min_hr_samples);       

    for i = 1:length(v_start_idx)
                
        % The things below can be written to a function if desired??
        rel_ppg = v_ppg_pre(v_start_idx(i):v_end_idx(i));
        rel_time = tr_ppg_pre(v_start_idx(i):v_end_idx(i));
        
        % Check whether the epoch can be extended by 2 s on both
        % sides --> not possible if it is the end of the epoch or
        % the start
        if v_start_idx(i)<2*fs_ppg || v_end_idx(i)>length(v_ppg_pre)-2*fs_ppg
            continue            % for now skip these epoch since these are probably an artifact (see example) but this could be relevant to check later on!
        end

        rel_ppg_spwvd = v_ppg_pre(v_start_idx(i)-fs_ppg*2:v_end_idx(i)+fs_ppg*2);           %extract epoch with two extra seconds at start and end to overcome boundary effects of the WVD/SPWVD function, necessary for the for loop --> also implemented for cwt to make it more or less the same!        
        hr_est = PPG_TFD_HR(rel_ppg_spwvd, tfd_length, MA, fs_ppg, kern_type, kern_params);           %wvd distribution does not require additional parameters for windowing
        
        %%-----Corresponding HR estimation time array-----&&
        if mod(length(rel_ppg),60)~=0           %if segment length is uneven --> substract fs 
            hr_time = rel_time(1:hr_est_samples:length(rel_ppg)-fs_ppg);
        else
            hr_time = rel_time(1:hr_est_samples:length(rel_ppg));
        end
        %%-------Save output-------%%
        t_epoch_unix = hr_time'*unix_ticks_ms + ts_sync;
        v_hr_ppg = [v_hr_ppg; hr_est];
        t_hr_unix = [t_hr_unix; t_epoch_unix];

    end
end

% Save the hr output in tsdf format
data_hr_est{1} = int32(t_hr_unix/unix_ticks_ms); % 32 bit integer and store it in unix seconds
data_hr_est{2} = single(v_hr_ppg);  % 32 bit float

location = "..\..\tests\data\5.quantification\ppg"; % Location to save the data
mkdir(location)
metafile_pre_template = metadata_list_sqa{time_idx_ppg}; % Copy the metadata template from the sqa time data --> most clean

if isempty (t_hr_unix)
    start_time_iso = datetime(0, "ConvertFrom", "posixtime", 'Format', 'dd-MMM-yyyy HH:mm:ss z', 'TimeZone', 'UTC');
    end_time_iso = datetime(0, "ConvertFrom", "posixtime", 'Format', 'dd-MMM-yyyy HH:mm:ss z', 'TimeZone', 'UTC');
else
    start_time_iso = datetime(t_hr_unix(1)/unix_ticks_ms, "ConvertFrom", "posixtime", 'Format', 'dd-MMM-yyyy HH:mm:ss z', 'TimeZone', 'UTC');
    end_time_iso = datetime(t_hr_unix(end)/unix_ticks_ms, "ConvertFrom", "posixtime", 'Format', 'dd-MMM-yyyy HH:mm:ss z', 'TimeZone', 'UTC');    % Convert the start and end time to ISO8601 format
end
metafile_pre_template.start_iso8601 = string(start_time_iso);
metafile_pre_template.end_iso8601 = string(end_time_iso);

metafile_time = metafile_pre_template;         % time vector metadata list
metafile_values_hr = metafile_pre_template;

metafile_time.channels = {'time'};
metafile_time.units = {'time_absolute_unix_s'};
metafile_time.file_name = 'hr_est_time.bin';

metafile_values_hr.channels = {'HR estimates'};
metafile_values_hr.units = {'min^-1'};
metafile_values_hr.freq_sampling_original = round(fs_ppg_est, 2); % Sampling rate in Hz of the raw data
metafile_values_hr.file_name = 'hr_est_values.bin';


meta_class{1} = metafile_time;
meta_class{2} = metafile_values_hr;

mat_metadata_file_name = "hr_est_meta.json";
save_tsdf_data(meta_class, data_hr_est, location, mat_metadata_file_name);