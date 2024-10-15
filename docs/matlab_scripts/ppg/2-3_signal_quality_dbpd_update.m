%% Main script to perform feature extraction for the signal quality assessment
% This script uses both PPG and accelerometer and performs the following
% steps:
%   1. Loading the preprocessed PPG and accelerometer data from the
%   implementation of the paradigma toolbox
%   2. Feature extraction
%   3. Storing the features


% Architecture overview
% The script implements the following steps:
%   2. IMU and PPG feature extraction
%   3. Signal quality assessment


%% Initalization
% Setting data paths + extracting metafilenames already
clearvars; close all; clc
addpath(genpath('..\..\..\..\paradigma-toolbox'))       % Add git repository to the path
addpath(genpath("..\..\..\..\tsdf4matlab"))       % Add wrapper to the path

unix_ticks_ms = 1000.0;
fs_ppg = 30;     % Establish the sampling rate desired for resampling PPG --> now chosen to be fixed on 30 Hz
fs_imu = 100;    % Establish the sampling rate desired for resampling IMU --> now chosen to be fixed on 30 Hz

raw_data_root = '..\..\..\tests\data\2.preprocessed_data\PPG\';
ppp_data_path_ppg = [raw_data_root 'PPG_new\'];
ppp_data_path_imu = [raw_data_root 'IMU_new\'];

%% 1. Loading all metadata of PPG and IMU
meta_ppg = tsdf_scan_meta(ppp_data_path_ppg);            % tsdf_scan_meta returns metafile struct containing information of all metafiles from all patients in tsdf_dirlist
meta_imu = tsdf_scan_meta(ppp_data_path_imu);

%% 2. Loading the preprocessed PPG and accelerometer data
meta_path_ppg = meta_ppg.tsdf_meta_fullpath;
meta_path_imu = meta_imu.tsdf_meta_fullpath;
[metadata_list_ppg, data_list_ppg] = load_tsdf_metadata_from_path(meta_path_ppg);
[metadata_list_imu, data_list_imu] = load_tsdf_metadata_from_path(meta_path_imu);

time_idx_ppg = tsdf_values_idx(metadata_list_ppg, 'time');
tr_ppg_pre = data_list_ppg{time_idx_ppg};
values_idx_ppg = tsdf_values_idx(metadata_list_ppg, 'samples');
v_ppg_pre = data_list_ppg{values_idx_ppg};

time_idx_imu = tsdf_values_idx(metadata_list_imu, 'time');
tr_imu_pre = data_list_imu{time_idx_ppg};
values_idx_imu = tsdf_values_idx(metadata_list_imu, 'samples');
v_imu_pre = data_list_imu{values_idx_imu};

%% 3. Feature extraction
% Create loop for 6s epochs with 5s overlap
epoch_length = 6; % in seconds
overlap = 5; % in seconds

% Number of samples in epoch
samples_per_epoch_ppg = epoch_length * fs_ppg;
samples_per_epoch_acc = epoch_length * fs_imu;

% Calculate number of samples to shift for each epoch
samples_shift_ppg = (epoch_length - overlap) * fs_ppg;
samples_shift_acc = (epoch_length - overlap) * fs_imu;          % Hoe krijg ik mijn segmenten precies gelijk zodat het niet toevallig een error geeft dat 1 langer is dan de ander

pwelchwin_acc = 3*fs_imu; % window length for pwelch
pwelchwin_ppg = 3*fs_ppg; % window length for pwelch
noverlap_acc = 0.5*pwelchwin_acc; % overlap for pwelch
noverlap_ppg = 0.5*pwelchwin_ppg; % overlap for pwelch

f_bin_res = 0.05;   % the treshold is set based on this binning --> so range of 0.1 Hz for calculating the PSD feature 
nfft_ppg = 0:f_bin_res:fs_ppg/2; % frequency bins for pwelch ppg
nfft_acc = 0:f_bin_res:fs_imu/2; % frequency bins for pwelch imu

features_ppg_scaled = [];
feature_acc = [];
t_unix_feat_total = [];
count = 0;
acc_idx = 1;

% Load classifier with corresponding mu and sigma to z-score the features
load("LR_model.mat")
mu = LR_model.mu; 
sigma = LR_model.sigma;
classifier = LR_model.classifier; % this classifier is only used later at the stage 7 

% DESCRIBE THE LOOPING OVER 6s SEGMENTS FOR BOTH PPG AND IMU AND CALCULATE FEATURES
for i = 1:samples_shift_ppg:(length(v_ppg_pre) - samples_per_epoch_ppg + 1)
        
        if acc_idx + samples_per_epoch_acc - 1 > length(v_acc_pre) % For the last epoch, check if the segment for IMU is too short (not 6 seconds), add a check for the last epoch first (if i == last then...)
                break
        else
            acc_segment = v_acc_pre(acc_idx:(acc_idx+samples_per_epoch_acc-1),:); % Extract the IMU window (6seconds)
        end
        ppg_segment = v_ppg_pre(i:(i + samples_per_epoch_ppg - 1)); % Extract the PPG window (6seconds)

        count = count + 1;
  
        %%--------Feature extraction + scaling--------%%
        % calculate features using ppg_features.m
        features = ppg_features(ppg_segment, fs_ppg);  % for now PPG_segment --> name can be adjusted!
        
        % Scaling using z-score 
        features_ppg_scaled(count,:) = normalize(features, 'center', mu, 'scale', sigma);

        % Calculating psd (power spectral density) of imu and ppg
        % in the Gait pipeline this is done using the Fourier transformation
        [pxx1,f1] = pwelch(acc_segment,hann(pwelchwin_acc), noverlap_acc, nfft_acc, fs_imu);
        PSD_imu = sum(pxx1,2);     % sum over the three axis
        [pxx2,f2] = pwelch(ppg_segment,hann(pwelchwin_ppg), noverlap_ppg, nfft_ppg, fs_ppg);
        PSD_ppg = sum(pxx2,1); % this does nothing, equal to PSD_ppg = pxx2
        
        % IMU feature extraction
        feature_acc(count,1) = acc_feature(f1, PSD_imu, f2, PSD_ppg); % Calculate the power ratio of the accelerometer signal in the PPG frequency range and store it as a row in the feature matrix (which has only 1 column)
        
        % time channel
        t_unix_feat_total(count,1) = tr_ppg_pre(i)*unix_ticks_ms + t_ppg(1);  % Save in absolute unix time ms
        acc_idx = acc_idx + samples_shift_acc; % update IMU_idx 
end
% at this point we have a matrix with features for PPG (10 column), IMU (1 columns), and a time vector

% THis is stored for later (stage 4)
v_sync_ppg_total(1,1) = ppg_indices(1); % start index --> needed for HR pipeline
v_sync_ppg_total(1,2) = ppg_indices(2); % end index --> needed for HR pipeline
v_sync_ppg_total(1,3) = segment_ppg(1);  % Segment index --> needed for HR pipeline --> in loop this is n
v_sync_ppg_total(1,4) = count; % Number of epochs in the segment --> needed for HR pipeline


%% 4a. Write TSDF PPG feature extraction output (ppg features)
location = "..\..\tests\data\3.extracted_features\ppg";
data_feat_ppg{1} = t_unix_feat_total;
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

%% 4b. Write TSDF PPG preprocessing output (accelerometer feature)
location = "..\..\tests\data\3.extracted_features\ppg";
data_feat_acc{1} = t_unix_feat_total;
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
%% 5. Classification


threshold_acc = 0.13; % Final threshold
[~, ppg_post_prob] = predict(classifier, features_ppg_scaled);       % Calculate posterior probability using LR model, python provides a similar function
ppg_post_prob_HQ = ppg_post_prob(:,1);
acc_label = feature_acc < threshold_acc; % logical (boolean) for not surpassing threshold_acc for imu feature. imu_label is one if we don't suspect the epoch to be disturbed by periodic movements! That is in line with 1 for HQ PPG

%% 7a. Storage of classification in tsdf, each element is a separate binary file (all but the 4th one include one column)
data_class{1} = int32(t_unix_feat_total/unix_ticks_ms); % 32 bit integer and store it in unix seconds
data_class{2} = single(ppg_post_prob_HQ);  % 32 bit float
data_class{3} = int8(acc_label); % 8 bit integer
data_class{4} = int32(v_sync_ppg_total); % 64 bit integer - 4 columns
data_class{5} = single(feature_acc); % 32 bit float

location = "..\..\tests\data\4.predictions\ppg";
metafile_pre_template = metadata_list_ppg{values_idx_ppg};

start_time_iso = datetime(t_unix_feat_total(1)/unix_ticks_ms, "ConvertFrom", "posixtime", 'Format', 'dd-MMM-yyyy HH:mm:ss z', 'TimeZone', 'UTC');
end_time_iso = datetime(t_unix_feat_total(end)/unix_ticks_ms, "ConvertFrom", "posixtime", 'Format', 'dd-MMM-yyyy HH:mm:ss z', 'TimeZone', 'UTC');    % Convert the start and end time to ISO8601 format

metafile_pre_template.start_iso8601 = string(start_time_iso);
metafile_pre_template.end_iso8601 = string(end_time_iso);
%metafile_pre_template.ppp_source_protobuf = "WatchData.IMU.Week104.raw";
% Contains both IMU and PPG original raw files for classification --> what
% is the most neat thing to do

metafile_time = metafile_pre_template;         % time vector metadata list
metafile_values_ppg = metafile_pre_template;
metafile_values_imu = metafile_pre_template; 
metafile_sync = metafile_pre_template;    
metafile_values_imu_feat = metafile_pre_template;

metafile_time.channels = {'time'};
metafile_time.units = {'time_absolute_unix_ms'};
metafile_time.file_name = 'classification_sqa_time.bin';

metafile_values_ppg.channels = {'post probability'};
metafile_values_ppg.units = {'probability'};
metafile_values_ppg.freq_sampling_original = fs_ppg_est; % Sampling rate in Hz
metafile_values_ppg.file_name = 'classification_sqa_ppg.bin';

metafile_values_imu.channels = {'accelerometer classification'};
metafile_values_imu.units = {'boolean_num'};
metafile_values_imu.freq_sampling_original = fs_imu_est; % Sampling rate in Hz
metafile_values_imu.file_name = 'classification_sqa_imu.bin';

metafile_sync.channels = {'ppg start index', 'ppg end index', 'ppg segment index', 'number of windows'};  % Define the channels
metafile_sync.units = {'index', 'index', 'index', 'none'};
metafile_sync.file_name = 'classification_sqa_sync.bin';

metafile_values_imu_feat.channels = {'Relative power'};
metafile_values_imu_feat.units = {'none'};
metafile_values_imu_feat.file_name = 'classification_sqa_feat_imu.bin';
metafile_values_imu_feat.bin_width = 2*f_bin_res;  % Save the bin width of the relative power ratio feature

meta_class{1} = metafile_time;
meta_class{2} = metafile_values_ppg;
meta_class{3} = metafile_values_imu;
meta_class{4} = metafile_sync;
meta_class{5} = metafile_values_imu_feat;

mat_metadata_file_name = "classification_sqa_meta.json";
save_tsdf_data(meta_class, data_class, location, mat_metadata_file_name)

%% 8. Incorporate HR estimation