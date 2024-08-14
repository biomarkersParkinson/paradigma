function [segment_ppg_total, segment_imu_total] = synchronization(ppg_meta, imu_meta)
% K.I. Veldkamp, PhD student AI4P, 29-02-24
% This function checks data availability between PPG and IMU and returns
% the synchronized indices

% Convert start and end times to datetime objects
ppg_start_time = datetime({ppg_meta.start_iso8601}', 'InputFormat', 'dd-MMM-yyyy HH:mm:ss Z', 'Format', 'yyyy-MM-dd''T''HH:mm:ss', 'TimeZone', 'UTC'); 
imu_start_time = datetime({imu_meta.start_iso8601}', 'InputFormat', 'dd-MMM-yyyy HH:mm:ss Z', 'Format', 'yyyy-MM-dd''T''HH:mm:ss', 'TimeZone', 'UTC');
ppg_end_time = datetime({ppg_meta.end_iso8601}', 'InputFormat', 'dd-MMM-yyyy HH:mm:ss Z', 'Format', 'yyyy-MM-dd''T''HH:mm:ss', 'TimeZone', 'UTC'); 
imu_end_time = datetime({imu_meta.end_iso8601}', 'InputFormat', 'dd-MMM-yyyy HH:mm:ss Z', 'Format', 'yyyy-MM-dd''T''HH:mm:ss', 'TimeZone', 'UTC');

% Create a time vector covering the entire range
time_vector_total = datetime(min([imu_start_time; ppg_start_time]), 'Format', 'yyyy-MM-dd HH:mm:ss'):seconds(1):datetime(max([imu_end_time; ppg_end_time]), 'Format', 'yyyy-MM-dd HH:mm:ss');

% Initialize variables
data_presence_ppg = zeros(size(time_vector_total));
data_presence_ppg_idx = zeros(size(time_vector_total));
data_presence_imu = zeros(size(time_vector_total));
data_presence_imu_idx = zeros(size(time_vector_total));

% Mark the segments of PPG data with 1
for i = 1:length(ppg_start_time)
    indices = time_vector_total >= ppg_start_time(i) & time_vector_total < ppg_end_time(i);
    data_presence_ppg(indices) = 1;
    data_presence_ppg_idx(indices) = i;
end

% Mark the segments of IMU data with 1
for i = 1:length(imu_start_time)
    indices = time_vector_total >= imu_start_time(i) & time_vector_total < imu_end_time(i);
    data_presence_imu(indices) = 1;
    data_presence_imu_idx(indices) = i;
end

% Find the indices where both PPG and IMU data are present
corr_indices = find(data_presence_ppg == 1 & data_presence_imu == 1);

% Find the start and end indices of each segment
corr_start_end = [];
start_idx = corr_indices(1);
for i = 2:length(corr_indices)    
    if corr_indices(i) - corr_indices(i-1) > 1
        end_idx = corr_indices(i-1);
        corr_start_end = [corr_start_end; start_idx, end_idx];
        start_idx = corr_indices(i);
    end
end

% Add the last segment
if ~isempty(corr_indices)
    corr_start_end = [corr_start_end;  start_idx, corr_indices(end)];
end

% Extract the synchronized indices for each segment
segment_ppg_total = [];
segment_imu_total = [];
for i = 1:size(corr_start_end,1)
    start_idx = corr_start_end(i,1);
    end_idx = corr_start_end(i,2);
    segment_ppg = unique(data_presence_ppg_idx(start_idx:end_idx))';
    segment_imu = unique(data_presence_imu_idx(start_idx:end_idx))';
    if length(segment_ppg) > 1 & length(segment_imu) == 1
        segment_ppg_total = [segment_ppg_total; segment_ppg];
        segment_imu_total = [segment_imu_total; segment_imu*ones(length(segment_ppg),1)];
    elseif length(segment_ppg) == 1 & length(segment_imu) > 1
        segment_ppg_total = [segment_ppg_total; segment_ppg*ones(length(segment_imu),1)];
        segment_imu_total = [segment_imu_total; segment_imu];
    elseif length(segment_ppg) == length(segment_imu)
        segment_ppg_total = [segment_ppg_total; segment_ppg];
        segment_imu_total = [segment_imu_total; segment_imu];
    else
        continue
    end
end

end