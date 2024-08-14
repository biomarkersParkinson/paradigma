function [IMU_filt, t_tar] = preprocessing_imu(t_orig, IMU_sample, fs_imu) 
%%----Preprocessing of the IMU pipeline----%%                        
    % Resampling to a uniform sampling rate at 30 Hz                                  
    t_tar = t_orig(1):1/fs_imu:t_orig(end);         % Correct target for resampling (by usage of t_start = t_orig(1) and t_end = t_orig(end));
    signal_resampled = spline(t_orig, IMU_sample', t_tar);
    
    % High-pass filter for detrending
    [b,a]=butter(4,0.2/(fs_imu/2),'high');  % high-pass filter for gravity removal
    IMU_filt = filtfilt(b,a,double(signal_resampled'));
   
   % [zhi,phi,khi] = butter(4,0.2,'high');
   % soshi = zp2sos(zhi,phi,khi);
   % IMU_filt2 = filtfilt(soshi,double(signal_resampled'));
end

