% Feature Extraction
% Estimate the mean absolute derivative values for a given IMU axes window
% #########################################################################
% Input:
% x - x-axis signal (array) - double
% y - y-axis signal (array) - double
% z - z-axis signal (array) - double
% Fs - Sampling frequency (scalar) - double
%
% #########################################################################
% Output:
% DerivativesOut - array - double
% [MeanAbsDx MeanAbsDy MeanAbsDz]: 1 x 3
% #########################################################################
function DerivativesOut = DerivativesExtract(x,y,z,Fs)

Ts = 1/Fs; % sampling step
dx = mean((1/Ts)*abs(diff(x,1,1))); % get the mean dx/dt
dy = mean((1/Ts)*abs(diff(y,1,1))); % get the mean dy/dt
dz = mean((1/Ts)*abs(diff(z,1,1))); % get the mean dz/dt

DerivativesOut = [dx dy dz]; % Defining output values