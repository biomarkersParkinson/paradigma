function  [PSDBandPowerOut] = PSDBandPower(Conf,Data,freq)
% #########################################################################
% Input
% Conf - Structure
%     .freqrange            - array - double - Frequency Range for feature Extraction
%     .StrAxis              - string - signal axis (or Sum) specified to be considered for feature extraction
%     .nfft                 - scalar - double - number of points for the
%                             FFT (number of samples)
%     .Fs                   - scalar - double - sampling Rate
%
% Data - PSD Matrix - matrix - double - N frequency values x (Nsignals + 1) - last column is the PSD sum (PSDx + PSDy + PSDz)
%
% freq - array - double - frequency vector associsted to the PSD;
%
%
%
% Output - Table with PSD features for the axis (or sum of the axis)
% selected
% PSDFeaturesOut - scalar - double (1 x 1)
%                    BandPower      - power in the range

% Get the estimation parameters
freqrange =  Conf.freqrange;
StrAxis   = Conf.StrAxis;
nfft      = Conf.nfft;
Fs        = Conf.Fs;

% Getting the signal axis or sum according to what specified in the input
switch StrAxis
    case 'X'
        Pxx = Data(:,1);
    case 'Y'
        Pxx = Data(:,2);
    case 'Z'
        Pxx = Data(:,3);
    case 'Sum'
        Pxx = Data(:,4);
end

f0 = Fs/nfft; % Spectral Resolution

FreqLowerBound   = freqrange(1); % lower frequency bound
FreqUpperBound   = freqrange(2); % upper frequency bound

% Total Power - area under PSD using an approximation by the left

InitialIndexBandForPower = find(freq == FreqLowerBound);      % Initial index for bandpower estimation
FinalIndexBandForPower   = find(freq == (FreqUpperBound-f0)); % Final index for bandpower estimation: area will be estimated using approximation by the left

BandPower      = f0*sum(Pxx(InitialIndexBandForPower:FinalIndexBandForPower));    % Power in the band specified

PSDBandPowerOut = BandPower; % Defining the output