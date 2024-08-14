function [PPG_data, t_tar] = preprocessing_ppg(t_orig, PPG_sample, fs_ppg)         
              
    % Resampling to a uniform sampling rate at 30 Hz                  
    t_tar = t_orig(1):1/fs_ppg:t_orig(end);         % Correct target for resampling (by usage of t_start and t_end);
    signal_resampled = spline(t_orig, PPG_sample, t_tar); % Spline interpolation used for resampling

    % Band-pass filter for detrending

    [b,a]=butter(4,[0.4, 3.5]/(fs_ppg/2),'bandpass');  % band-pass filter for detrending
    PPG_filt = filtfilt(b,a,double(signal_resampled)); % Apply filter
    PPG_data = PPG_filt';   % In case of a sample package error where PPG_filt consists of multiple "parts"

end
