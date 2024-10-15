function  [PSDFeaturesOut] = PSDExtrOpt(Conf,Data,freq)
% #########################################################################
% Input
% Conf - Structure
%     .freqrange            - array - double - Frequency Range for feature 
%                                              Extraction
%     .SpectralIntervalPeak - scalar - double - number of samples around
%                                               the peak to evaluate Fixed 
%                                               Dominant Power
%     .StrAxis              - string - signal axis (or Sum) specified to be
%                                      considered for feature extraction
%     .nfft                 - scalar - double - number of points for the
%                             FFT (number of samples)
%     .Fs                   - scalar - double - sampling Rate
%
% Data - PSD Matrix - matrix - double - Frequency values x (NPSDsignals + 1)
%                                       last column is the PSD sum 
%                                       (PSDx + PSDy + PSDz)
%
% freq - array - double - frequency vector associsted to the PSD;
% #########################################################################
% Output - Table with PSD features for the axis (or sum of the axis)
% selected
% PSDFeaturesOut - array - double (1 x 3)
%                     BandPower - power in the range
%                     FreqPeak  - dominant frequency in the tremor range
%                     FixedDomPower - dominant power 1.25 Hz around the 
%                                     tremor peak
% #########################################################################

% Get the estimation parameters
freqrange            =  Conf.freqrange;
SpectralIntervalPeak =  Conf.SpectralIntervalPeak;
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

SpectrumInterval = SpectralIntervalPeak; % Number of samples taken around 
                                         % the spectral peak to compute 
                                         % dominant power

FreqLowerBound   = freqrange(1); % lower frequency bound
FreqUpperBound   = freqrange(2); % upper frequency bound

% Logical array specifying the specified frequency
BandFreq    = (freq >= FreqLowerBound & freq <= FreqUpperBound);

% Power - area under PSD using an approximation by the left
% Dominant Power: 5 frequency coefficients are considered around the peak
% (2 coefficients for each side) implying a frequency width of 1.25 Hz

% Initial index for bandpower estimation
InitialIndexBandForPower = find(freq == FreqLowerBound);      

% Final index for bandpower estimation: area will be estimated using 
% approximation by the left
FinalIndexBandForPower   = find(freq == (FreqUpperBound-f0)); 

% Bandpower evaluation
BandPower      = f0*sum(Pxx(InitialIndexBandForPower:FinalIndexBandForPower));

% Dominanting Frequency Characteristics
[max_power, ind_max_rel]  = max(Pxx(BandFreq));           % Peak values and index within the specified band
indplus                   = find(BandFreq == 1);          % Initial index of the band specified
ind_max                   = indplus(1) + ind_max_rel - 1; % index of the dominant frequency

FreqPeak                  = freq(ind_max); % Dominant frequency

LowerFixedPeakBound    = max(1,ind_max - SpectrumInterval); % lower bound frequency index avoiding negative indexed for dominant frequency lower than 0.5 Hz
MaxFixedPeakBound      = ind_max       + SpectrumInterval;  % upper bound frequency index

% Dominant power (area under PSD using approximation by the left)
FixedDomPower          = f0*(sum(Pxx(LowerFixedPeakBound:MaxFixedPeakBound))); % this takes 5 PSD coefficients which implies a bandwidth of 1.25 Hz

% Dominant power normalized by the band power
% FixedDomPowerRatio = FixedDomPower/BandPower;

% Concatening features:
PxxFeatures = [BandPower FreqPeak FixedDomPower];

PSDFeaturesOut = PxxFeatures; % Defining the output