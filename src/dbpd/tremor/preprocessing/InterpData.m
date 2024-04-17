% Interpolate Data
% Diogo C. Soriano
% 15/08/2022

% #########################################################################
% Input:
% tr            - array  - double - running time in (s) after transformation from Unix time;
% y_curr_gyro   - matrix - double - gyroscope matrix Nsamples x 3: columns [x,y,z]
% Fs            - scalar - double - sampling rate (Hz);
% unix_ticks_ms - scalar - double - Unix ticks / ms
% ts            - scalar - double - Unix time start

% Output:
% t_imu_proc{n} - cell - (double) - Unix time corrected (interpolated);
% v_imu_proc{n} - cell - (double) - Nsamples x Nchannels - Gyroscope
%                                   signals interpolated. Columns: [x y z]
% #########################################################################
function [tcorrected,y_curr_gyro_interp] = InterpData(tr,y_curr_gyro,Fs,unix_ticks_ms,ts)

[tunique, indunique] = unique(tr);

y_curr_gyro = y_curr_gyro(indunique,:);

ti = (0:1/Fs:tunique(end))';

if length(y_curr_gyro(:,1)) > 3
    
    y_curr_gyro_interp = interp1(tunique,y_curr_gyro,ti,'spline');  % Use splines to interpolate the data
        
end

tcorrected = ti*unix_ticks_ms + ts;
