function [ppg_indices, imu_indices] = extract_overlapping_segments(t_unix_ppg, t_unix_imu)
    % K.I. Veldkamp, PhD student AI4P, 29-02-24
    % Function to extract indices indicating overlapping data between IMU and
    % PPG segment
    
        % Convert UNIX milliseconds to seconds
        ppg_time = t_unix_ppg / 1000; % Convert milliseconds to seconds
        imu_time = t_unix_imu / 1000; % Convert milliseconds to seconds
    
        % Determine the overlapping time interval
        start_time = max(ppg_time(1), imu_time(1));
        end_time = min(ppg_time(end), imu_time(end));
    
        % Convert overlapping time interval to indices
        ppg_start_index = find(ppg_time >= start_time, 1);
        ppg_end_index = find(ppg_time <= end_time, 1, 'last');
        imu_start_index = find(imu_time >= start_time, 1);
        imu_end_index = find(imu_time <= end_time, 1, 'last');
    
        % Extract overlapping segments
        ppg_indices = [ppg_start_index, ppg_end_index];
        imu_indices = [imu_start_index, imu_end_index];
    end