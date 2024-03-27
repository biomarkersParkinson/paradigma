function  [PSDFeaturesOut] = PSDExtrAxis(Conf,Data,freq)
% #########################################################################
% Input
% Conf - Structure
%     .freqrange            - array  - double - frequency Range for 
%                                               dominant power evaluation
%     .SpectralIntervalPeak - scalar - double - number of samples around
%                                               the peak to evaluate 
%                                               Fixed Dominant Power
%     .StrAxis              - string - signal axis (or Sum) specified to be
%                                      considered for feature extraction
%     .nfft                 - scalar - double - number of points for the
%                             FFT (number of samples)
%     .Fs                   - scalar - double - sampling Rate
%     .MinFreqOverallBand   - scalar - double - minimum frequency for 
%                                               overall power evaluation
%     .MaxFreqOverallBand   - scalar - double - maximum frequency for 
%                                               overall power evaluation
%
% Data - PSD Matrix - matrix - double - N frequency values x 
%                                       (Nsignals + 1) - last column is the
%                                       PSD sum (PSDx + PSDy + PSDz)
%
% freq - array - double - frequency vector associsted to the PSD;
% #########################################################################
% Output
% PSDFeaturesOut - matrix - double (1 x 2)
%                    (1) PowerAxis - power in the full range 0.5 - 25 Hz
%                    (2) DomPower  - dominant power 1.25 Hz around the peak
%                                    within the range: freqrange
% #########################################################################

% Get the estimation parameters
freqrange =  Conf.freqrange;
SpectralIntervalPeak =  Conf.SpectralIntervalPeak;
StrAxis   = Conf.StrAxis;
nfft      = Conf.nfft;
Fs        = Conf.Fs;
MinFreqOverallBand = Conf.MinFreqOverallBand;
MaxFreqOverallBand = Conf.MaxFreqOverallBand;

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

SpectrumInterval = SpectralIntervalPeak; % Number of samples taken around the spectral peak to compute dominant power
FreqLowerBound   = freqrange(1); % lower frequency bound
FreqUpperBound   = freqrange(2); % upper frequency bound

% Logical array specifying the specified frequency
BandFreq    = (freq >= FreqLowerBound & freq <= FreqUpperBound);

% Total Power - area under PSD using an approximation by the left: 5
% frequency coefficients are considered for symmetry around the peak
% implying a frequency width of 1.25 Hz

InitialIndexFullBandPower  = find(freq == MinFreqOverallBand);         % Initial index for full-band power estimation - this is required to remove frequency components associated with gyroscope drift
FinalIndexFullBandPower    = find(freq  == (MaxFreqOverallBand-f0));   % Final index for full-band power estimation: area will be estimated using approximation by the left

TotalPower     = f0*sum(Pxx(InitialIndexFullBandPower:FinalIndexFullBandPower)); % Total power up to 25 Hz

% Dominanting Frequency Characteristics
[max_power, ind_max_rel]  = max(Pxx(BandFreq));           % Peak values and index within the specified band
indplus                   = find(BandFreq == 1);          % Initial index of the band specified
ind_max                   = indplus(1) + ind_max_rel - 1; % index of the dominant frequency

FreqPeak                  = freq(ind_max); % Dominant frequency

LowerFixedPeakBound    = max(1,ind_max - SpectrumInterval); % lower bound frequency index avoiding negative indexed for dominant frequency lower than 0.5 Hz
MaxFixedPeakBound      = ind_max       + SpectrumInterval;  % upper bound frequency index

% Dominant power (area under PSD using approximation by the left)
FixedDomPower          = f0*(sum(Pxx(LowerFixedPeakBound:MaxFixedPeakBound))); % this takes 5 PSD coefficients which implies a bandwidth of 1.25 Hz

% Concatening features:
PxxFeatures = [TotalPower FixedDomPower];

PSDFeaturesOut = PxxFeatures; % Defining the output
