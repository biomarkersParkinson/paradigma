function [ppg_indices, imu_indices] = extract_overlapping_segments(ts_ppg, ts_imu, t_unix_ppg, t_unix_imu)
% K.I. Veldkamp, PhD student AI4P, 29-02-24
% Function to extract indices indicating overlapping data between IMU and
% PPG segment
    % Convert Unix timestamps to absolute time
    ppg_absolute_time = datetime(ts_ppg / 1000, 'ConvertFrom', 'posixtime');
    imu_absolute_time = datetime(ts_imu / 1000, 'ConvertFrom', 'posixtime');

    % Convert UNIX milliseconds to seconds
    ppg_time = t_unix_ppg / 1000; % Convert milliseconds to seconds
    imu_time = t_unix_imu / 1000; % Convert milliseconds to seconds

    % Determine the overlapping time interval
    start_time = max(ppg_absolute_time, imu_absolute_time);
    end_time = min(ppg_absolute_time + seconds(ppg_time(end) - ppg_time(1)), imu_absolute_time + seconds(imu_time(end)-imu_time(1)));

    % Convert overlapping time interval to indices
    ppg_start_index = find(ppg_time >= posixtime(start_time), 1);
    ppg_end_index = find(ppg_time <= posixtime(end_time), 1, 'last');
    imu_start_index = find(imu_time >= posixtime(start_time), 1);
    imu_end_index = find(imu_time <= posixtime(end_time), 1, 'last');

    % Extract overlapping segments
    ppg_indices = [ppg_start_index, ppg_end_index];
    imu_indices = [imu_start_index, imu_end_index];
end