%% Main script to perform signal quality assessment of wearable PPG
% This script uses both PPG and accelerometer and performs the following
% steps:
%   1. Loading all metadata of PPG and IMU
%   2. Query on data availability + synchronization
%   3. Loading relevant segment sensor data using tsdf wrapper (start for loop over synchronized segment indices)
%   4. Synchronize the data (correct indices etc)
%   5. Data preprocessing
%   6. Feature extraction
%   7. Classification


%% Initalization
% Setting data paths + extracting metafilenames already
clear all; close all; clc
addpath(genpath('..\..\..\PPG_pipeline'))       % Add git repository to the path
addpath(genpath("..\..\..\\tsdf4matlab"))       % Add wrapper to the path

unix_ticks_ms = 1000.0;
fs_ppg = 30;     % Establish the sampling rate desired for resampling PPG --> now chosen to be fixed on 30 Hz
fs_imu = 100;    % Establish the sampling rate desired for resampling IMU --> now chosen to be fixed on 30 Hz

raw_data_root = '..\..\tests\data\1.sensor_data\';
ppp_data_path_ppg = [raw_data_root 'PPG\'];
meta_segments_list_ppg = dir(fullfile(ppp_data_path_ppg, 'PPG_meta.json'));   % create the segment list

ppp_data_path_imu = [raw_data_root 'IMU\'];
meta_segments_list_imu = dir(fullfile(ppp_data_path_imu, 'IMU_meta.json'));   % create the segment list

meta_filenames_ppg = {meta_segments_list_ppg.name}; % get names
meta_filenames_imu = {meta_segments_list_imu.name}; % get names

%% 1. Loading all metadata of PPG and IMU
meta_ppg = tsdf_scan_meta(ppp_data_path_ppg);            % tsdf_scan_meta returns metafile struct containing information of all metafiles from all patients in tsdf_dirlist
meta_imu = tsdf_scan_meta(ppp_data_path_imu);

%% 2. Query on data availability + synchronization
[segment_ppg, segment_imu] = synchronization(meta_ppg, meta_imu);  % PPG segment and IMU segment indices corresponding to eachother (where there is overlapping data)
% NOT NEEDED FOR TEST DATA BUT SHOULD BE FUNCTIONALITY IN THE TOOLBOX

%% 3. Loading relevant segment sensor data using tsdf wrapper --> TO BE ADJUSTED
%%--------Load PPG + IMU data-------%%
n_files_sync = length(segment_ppg); % For test data this is 1 for final application it can be anything and it requires looping through the script

% for n = 1:n_files_sync --> if there are more segments!!
% end

n = 1;
meta_path_ppg = meta_ppg(segment_ppg{n}).tsdf_meta_fullpath;
meta_path_imu = meta_imu(segment_imu{n}).tsdf_meta_fullpath;

[metadata_list_ppg, data_list_ppg] = load_tsdf_metadata_from_path(meta_path_ppg);
[metadata_list_imu, data_list_imu] = load_tsdf_metadata_from_path(meta_path_imu);

time_idx_ppg = tsdf_values_idx(metadata_list_ppg, 'time');    % added for correctness instead of assuming that the idx is the same for every time we use load_tsdf_metadata_from_path --> or is this unnecessary --> assumption it is different for PPG and IMU!!
time_idx_imu = tsdf_values_idx(metadata_list_imu, 'time');
values_idx_ppg = tsdf_values_idx(metadata_list_ppg, 'samples');
values_idx_imu = tsdf_values_idx(metadata_list_imu, 'samples');

t_iso_ppg = metadata_list_ppg{time_idx_ppg}.start_iso8601;
t_iso_imu = metadata_list_imu{time_idx_imu}.start_iso8601;

datetime_ppg = datetime(t_iso_ppg, 'InputFormat', 'dd-MMM-yyyy HH:mm:ss Z', 'TimeZone', 'UTC');
datetime_imu = datetime(t_iso_imu, 'InputFormat', 'dd-MMM-yyyy HH:mm:ss Z', 'TimeZone', 'UTC');

t_diff_ppg = data_list_ppg{time_idx_ppg};
t_diff_imu = data_list_imu{time_idx_imu};

ts_ppg = posixtime(datetime_ppg) * unix_ticks_ms;      % calculate the unix timestamp in ms
ts_imu = posixtime(datetime_imu) * unix_ticks_ms;      % calculate the unix timestamp in ms

t_ppg = cumsum(double(data_list_ppg{time_idx_ppg})) + ts_ppg(n);
t_imu = cumsum(double(data_list_imu{time_idx_imu})) + ts_imu(n);

tr_ppg = (t_ppg-ts_ppg)/unix_ticks_ms;
tr_imu = (t_imu-ts_imu)/unix_ticks_ms;

v_ppg = data_list_ppg{values_idx_ppg};
v_imu = data_list_imu{values_idx_imu};             % store data values for every seperate tsdf file in cell
scale_factors = metadata_list_imu{values_idx_imu}.scale_factors';

%% 4. Data synchronization on right indices
fs_ppg_est = 1000/median(t_diff_ppg); 
fs_imu_est = 1000/median(t_diff_imu);
[ppg_indices, imu_indices] = extract_overlapping_segments(t_ppg, t_imu);

%%---Update data vectors on synchronized labels---%%
v_ppg = v_ppg(ppg_indices(1):ppg_indices(2));
v_imu = v_imu(imu_indices(1):imu_indices(2),:);
t_ppg = t_ppg(ppg_indices(1):ppg_indices(2));
t_imu = t_imu(imu_indices(1):imu_indices(2));
tr_ppg = tr_ppg(ppg_indices(1):ppg_indices(2));
tr_imu = tr_imu(imu_indices(1):imu_indices(2));

ts_sync = ts_ppg + tr_ppg(1)*unix_ticks_ms;   % update ts_sync by the first relative time point containing both PPG and IMU a

tr_ppg = tr_ppg - tr_ppg(1);   % update tr_ppg by the first relative time point containing both PPG and IMU --> should be done after ts_sync is updated
tr_imu = tr_imu - tr_imu(1);  % update tr_imu by the first relative time point containing both PPG and IMU


%% 5. Data preprocessing
%%--Preprocessing both IMU and PPG%%  
v_acc_scaled = scale_factors(1,1:3).*double(v_imu(:,1:3));     % Extract only the accelerometer channels and multiply them using scale factors! --> now based on indices but preferably on channel names in metadata???
min_window_length = 30;

%%----NEEDS TO BE IMPLEMENTED IN FUNCTION!!--%%
if length(v_ppg) < fs_ppg * min_window_length || length(v_acc_scaled) < fs_imu * min_window_length    % Only resample, feature calculation and classification on arrays > 30s since these are required for HR(V) analysis later on --> maybe add this to the synchronization  
    warning('Sample is of insufficient length!')
else
    [v_ppg_pre, tr_ppg_pre] = preprocessing_ppg(tr_ppg, v_ppg, fs_ppg);   % call preprocessing_ppg.m function to preprocess every segment seperately
    [v_acc_pre, tr_acc_pre] = preprocessing_imu(tr_imu, v_acc_scaled, fs_imu);   % List of call preprocessing_imu.m function to preprocess every segment seperately
end

%% 5a. Write TSDF PPG preprocessing output
location = "..\..\tests\data\2.preprocessed_data\ppg";
data_pre_ppg{1} = tr_ppg_pre;
data_pre_ppg{2} = v_ppg_pre;

metafile_pre_template = metadata_list_ppg{values_idx_ppg};

start_time_iso = datetime(ts_sync/unix_ticks_ms, "ConvertFrom", "posixtime", 'Format', 'dd-MMM-yyyy HH:mm:ss z', 'TimeZone', 'UTC');
end_time_iso = datetime((ts_sync+tr_ppg_pre(end)*1000)/unix_ticks_ms, "ConvertFrom", "posixtime", 'Format', 'dd-MMM-yyyy HH:mm:ss z', 'TimeZone', 'UTC');

metafile_pre_template.start_iso8601 = string(start_time_iso);
metafile_pre_template.end_iso8601 = string(end_time_iso);
metafile_pre_template.ppp_source_protobuf = "WatchData.PPG.Week104.raw";

metafile_time = metafile_pre_template;         % time vector metadata list as a template and adjust it
metafile_values = metafile_pre_template;

metafile_time.channels = {'time'};
metafile_time.units = {'relative_ms'};
metafile_time.freq_sampling_original = fs_ppg_est;
metafile_time.file_name = 'PPG_time.bin';

metafile_values.channels = {'green'};
metafile_values.units = {'none'};
metafile_values.freq_sampling_original = fs_ppg_est;
metafile_values.file_name = 'PPG_samples.bin';

meta_pre_ppg{1} = metafile_time;
meta_pre_ppg{2} = metafile_values;
mat_metadata_file_name = "PPG_meta.json";
save_tsdf_data(meta_pre_ppg, data_pre_ppg, location, mat_metadata_file_name)

%% 5b. Write TSDF PPG preprocessing output
data_pre_acc{1} = tr_acc_pre;
data_pre_acc{2} = v_acc_pre;

metafile_pre_template = metadata_list_ppg{values_idx_ppg};

start_time_iso = datetime(ts_sync/unix_ticks_ms, "ConvertFrom", "posixtime", 'Format', 'dd-MMM-yyyy HH:mm:ss z', 'TimeZone', 'UTC');
end_time_iso = datetime((ts_sync+tr_acc_pre(end)*unix_ticks_ms)/unix_ticks_ms, "ConvertFrom", "posixtime", 'Format', 'dd-MMM-yyyy HH:mm:ss z', 'TimeZone', 'UTC');

metafile_pre_template.start_iso8601 = string(start_time_iso);
metafile_pre_template.end_iso8601 = string(end_time_iso);
metafile_pre_template.ppp_source_protobuf = "WatchData.IMU.Week104.raw";

metafile_time = metafile_pre_template;         % time vector metadata list
metafile_values = metafile_pre_template;

metafile_time.channels = {'time'};
metafile_time.units = {'ms'};
metafile_time.freq_sampling_original = fs_imu_est;
metafile_time.file_name = 'acceleration_time.bin';

metafile_values.channels = {'acceleration_x', 'acceleration_y', 'acceleration_z'};
metafile_values.units = {'m/s/s', 'm/s/s', 'm/s/s'};
metafile_values.freq_sampling_original = fs_imu_est; % Sampling rate in Hz
metafile_values.file_name = 'acceleration_samples.bin';

meta_pre_acc{1} = metafile_time;
meta_pre_acc{2} = metafile_values;
mat_metadata_file_name = "acceleration_meta.json";
save_tsdf_data(meta_pre_acc, data_pre_acc, location, mat_metadata_file_name)
%% 6. Feature extraction
% Create loop for 6s epochs with 5s overlap
count = 1;
epoch_length = 6; % in seconds
overlap = 5; % in seconds

% Number of samples in epoch
samples_per_epoch_ppg = epoch_length * fs_ppg;
samples_per_epoch_acc = epoch_length * fs_imu;

% Calculate number of samples to shift for each epoch
samples_shift_ppg = (epoch_length - overlap) * fs_ppg;
samples_shift_acc = (epoch_length - overlap) * fs_imu;          % Hoe krijg ik mijn segmenten precies gelijk zodat het niet toevallig een error geeft dat 1 langer is dan de ander

pwelchwin_acc = 3*fs_imu;
pwelchwin_ppg = 3*fs_ppg;
noverlap_acc = 0.5*pwelchwin_acc;
noverlap_ppg = 0.5*pwelchwin_ppg;

f_bin_res = 0.05;   % the treshold is set based on this binning --> so range of 0.1 Hz for calculating the PSD feature 
nfft_ppg = 0:f_bin_res:fs_ppg/2;
nfft_acc = 0:f_bin_res:fs_imu/2;

features_ppg_scaled = [];
feature_acc = [];
t_unix_feat_ppg = [];
t_unix_feat_acc = [];
acc_idx = 1;

% Load classifier with corresponding mu and sigma to z-score the features
load("LR_model.mat")
mu = LR_model.mu;
sigma = LR_model.sigma;
classifier = LR_model.classifier;

% DESCRIBE THE LOOPING OVER 6s SEGMENTS FOR BOTH PPG AND IMU AND CALCULATE FEATURES
for i = 1:samples_shift_ppg:(length(v_ppg_pre) - samples_per_epoch_ppg + 1)
    
        ppg_segment = v_ppg_pre(i:(i + samples_per_epoch_ppg - 1));
        acc_segment = v_acc_pre(acc_idx:(acc_idx+samples_per_epoch_acc-1),:);
  
        %%--------Feature extraction + scaling--------%%
        % calculate features using Features_final.m
        features = ppg_features(ppg_segment, fs_ppg);  % for now PPG_segment --> name can be adjusted!
        
        % Scaling using z-score 
        features_ppg_scaled(count,:) = normalize(features, 'center', mu, 'scale', sigma);

        % Calculating psd of imu and ppg
        [pxx1,f1] = pwelch(acc_segment,hann(pwelchwin_acc), noverlap_acc, nfft_acc, fs_imu);
        PSD_imu = sum(pxx1,2);     % sum over the three axis
        [pxx2,f2] = pwelch(ppg_segment,hann(pwelchwin_ppg), noverlap_ppg, nfft_ppg, fs_ppg);
        PSD_ppg = sum(pxx2,1);
        
        feature_acc(count,1) = acc_feature(f1, PSD_imu, f2, PSD_ppg);
        
        t_unix_feat_ppg(count,1) = tr_ppg_pre(i)*unix_ticks_ms + t_ppg(1);  % Save in absolute unix time ms
        t_unix_feat_acc(count,1) = tr_acc_pre(acc_idx)*unix_ticks_ms + t_imu(1);

        acc_idx = acc_idx + samples_shift_acc; % update IMU_idx 
        count = count + 1;
end

t_epochs_start = t_unix_feat_ppg(1);
t_epochs_end = t_unix_feat_ppg(1) + tr_ppg_pre(i+samples_shift_ppg*overlap-1)*unix_ticks_ms;


%% 6a. Write TSDF PPG feature extraction output (ppg features)
location = "..\..\tests\data\3.extracted_features\ppg";
data_feat_ppg{1} = t_unix_feat_ppg;
data_feat_ppg{2} = single(features_ppg_scaled);

metafile_pre_template = metadata_list_ppg{values_idx_ppg};

start_time_iso = datetime(ts_sync/unix_ticks_ms, "ConvertFrom", "posixtime", 'Format', 'dd-MMM-yyyy HH:mm:ss z', 'TimeZone', 'UTC');
end_time_iso = datetime((ts_sync+tr_acc_pre(end)*unix_ticks_ms)/unix_ticks_ms, "ConvertFrom", "posixtime", 'Format', 'dd-MMM-yyyy HH:mm:ss z', 'TimeZone', 'UTC');

metafile_pre_template.start_iso8601 = string(start_time_iso);
metafile_pre_template.end_iso8601 = string(end_time_iso);
metafile_pre_template.ppp_source_protobuf = "WatchData.PPG.Week104.raw";

metafile_time = metafile_pre_template;         % time vector metadata list as a template and adjust it
metafile_values = metafile_pre_template;

metafile_time.channels = {'time'};
metafile_time.units = {'time_absolute_unix_ms'};
metafile_time.freq_sampling_original = fs_ppg_est;
metafile_time.file_name = 'features_ppg_time.bin';

metafile_values.channels = {'variance', 'mean', 'median', 'kurtosis', 'skewness', 'dominant frequency', 'relative power', 'spectral entropy', 'SNR', 'correlation peak'};
metafile_values.units = {'none', 'none', 'none', 'none', 'none', 'Hz', 'none', 'none', 'none', 'none'};
metafile_values.freq_sampling_original = fs_ppg_est;
metafile_values.file_name = 'features_ppg_samples.bin';

meta_feat_ppg{1} = metafile_time;
meta_feat_ppg{2} = metafile_values;
mat_metadata_file_name = "features_ppg_meta.json";
save_tsdf_data(meta_feat_ppg, data_feat_ppg, location, mat_metadata_file_name)

%% 6b. Write TSDF PPG preprocessing output (accelerometer feature)
location = "..\..\tests\data\3.extracted_features\ppg";
data_feat_acc{1} = t_unix_feat_acc;
data_feat_acc{2} = single(feature_acc);
metafile_pre_template = metadata_list_ppg{values_idx_ppg};

start_time_iso = datetime(ts_sync/unix_ticks_ms, "ConvertFrom", "posixtime", 'Format', 'dd-MMM-yyyy HH:mm:ss z', 'TimeZone', 'UTC');
end_time_iso = datetime((ts_sync+tr_acc_pre(end)*unix_ticks_ms)/unix_ticks_ms, "ConvertFrom", "posixtime", 'Format', 'dd-MMM-yyyy HH:mm:ss z', 'TimeZone', 'UTC');

metafile_pre_template.start_iso8601 = string(start_time_iso);
metafile_pre_template.end_iso8601 = string(end_time_iso);
metafile_pre_template.ppp_source_protobuf = "WatchData.IMU.Week104.raw";

metafile_time = metafile_pre_template;         % time vector metadata list
metafile_values = metafile_pre_template;

metafile_time.channels = {'time'};
metafile_time.units = {'time_absolute_unix_ms'};
metafile_time.freq_sampling_original = fs_imu_est;
metafile_time.file_name = 'feature_acc_time.bin';

metafile_values.channels = {'relative power acc'};
metafile_values.units = {'none'};
metafile_values.freq_sampling_original = fs_imu_est; % Sampling rate in Hz
metafile_values.file_name = 'feature_acc_samples.bin';

meta_feat_acc{1} = metafile_time;
meta_feat_acc{2} = metafile_values;
mat_metadata_file_name = "feature_acc_meta.json";
save_tsdf_data(meta_feat_acc, data_feat_acc, location, mat_metadata_file_name)
%% 7. Classification
threshold_acc = 0.15; % to be determined!!
[~, ppg_post_prob] = predict(classifier, features_ppg_scaled);       % Calculate posterior probability using LR model 
ppg_post_prob_HQ = ppg_post_prob(:,1);
acc_label = feature_acc < threshold_acc; % logical (boolean) for not surpassing threshold_acc for imu feature. imu_label is one if we don't suspect the epoch to be disturbed by periodic movements! That is in line with 1 for HQ PPG
%% 7a. Storage of classification in tsdf
data_class{1} = t_unix_feat_ppg;
data_class{2} = single(ppg_post_prob_HQ);  % 32 bit float
data_class{3} = int8(acc_label);

location = "..\..\tests\data\4.predictions\ppg";
metafile_pre_template = metadata_list_ppg{values_idx_ppg};

start_time_iso = datetime(ts_sync/unix_ticks_ms, "ConvertFrom", "posixtime", 'Format', 'dd-MMM-yyyy HH:mm:ss z', 'TimeZone', 'UTC');
end_time_iso = datetime((ts_sync+tr_acc_pre(end)*unix_ticks_ms)/unix_ticks_ms, "ConvertFrom", "posixtime", 'Format', 'dd-MMM-yyyy HH:mm:ss z', 'TimeZone', 'UTC');

metafile_pre_template.start_iso8601 = string(start_time_iso);
metafile_pre_template.end_iso8601 = string(end_time_iso);
%metafile_pre_template.ppp_source_protobuf = "WatchData.IMU.Week104.raw";
% Contains both IMU and PPG original raw files for classification --> what
% is the most neat thing to do

metafile_time = metafile_pre_template;         % time vector metadata list
metafile_values_ppg = metafile_pre_template;
metafile_values_imu = metafile_pre_template; 

metafile_time.channels = {'time'};
metafile_time.units = {'time_absolute_unix_ms'};
metafile_time.freq_sampling_original = fs_ppg_est;  % For now ppg, but it is not relevant right??
metafile_time.file_name = 'classification_sqa_time.bin';

metafile_values_ppg.channels = {'post probability'};
metafile_values_ppg.units = {'probability'};
metafile_values_ppg.freq_sampling_original = fs_ppg_est; % Sampling rate in Hz
metafile_values_ppg.file_name = 'classification_sqa_ppg.bin';

metafile_values_imu.channels = {'accelerometer classification'};
metafile_values_imu.units = {'boolean_num'};
metafile_values_imu.freq_sampling_original = fs_imu_est; % Sampling rate in Hz
metafile_values_imu.file_name = 'classification_sqa_imu.bin';

meta_class{1} = metafile_time;
meta_class{2} = metafile_values_ppg;
meta_class{3} = metafile_values_imu;

mat_metadata_file_name = "classification_sqa_meta.json";
save_tsdf_data(meta_class, data_class, location, mat_metadata_file_name)

%% 8. Incorporate HR estimation