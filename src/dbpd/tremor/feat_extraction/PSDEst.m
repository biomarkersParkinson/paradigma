% #########################################################################
% Input
% Conf - Configuration Strucutre
%     .pwelchwin - (scalar) - double - number of samples - window for Welch periodogram
%     .noverlap  - (scalar) - double - number of samples - ovserlap between windows
%     .Fs        - (scalar) - double - sampling frequency
%     .nfft      - (scalar) - double - number of samples for FFT evaluation
%     .freqmax   - (scalar) - double - maximal frequency of interest
%
% Data           - (matrix) - double - IMU time series (Nsamples x 3 columns - x y z arrays)

% Output
% PSDMatrix - (matrix) - double - (Nfreq x 4 columns) - PSD coefficients for x, y 
% and z and sum (fourth column: PSDx + PSDy + PSDz)  
%
% FreqVect  - array - double - Frequency array in Hz;
%
% FreqPeak  - scalar - double - Frequency (Hz) of the spectral peak
% #########################################################################

function [PSDMatrix,FreqVect,FreqPeak] = PSDEst(Conf,Data)

% Get PSD estimating parameters from the configuration structure
pwelchwin = Conf.pwelchwin;
noverlap  = Conf.noverlap;
Fs        = Conf.Fs;
nfft      = Conf.nfft;
freqmax   = Conf.freqmax;
freqmin   = Conf.freqmin;

% Get number of samples and number of signals to be manipulated
[Nsamples,Nsignals] = size(Data);

% Estimate the PSD coefficients (Pxx) for the frequencies (freqlong)  
% hann - Hanning window
[Pxx,freqlong] = pwelch(Data,hann(pwelchwin),noverlap,nfft,Fs);

% Finding the index of the maximal frequency to be considered
indcutfreq     = find(freqlong == freqmax);

% Get the frequency vector up to be maximal frequency
FreqVect       = freqlong(1:indcutfreq);

% Store the PSD coefficients for each IMU signal [x y z] up the maximal
% frequncy into a matrix (PSDMatrix)
PSDMatrix = Pxx(1:indcutfreq,:);

% Fourth column of PSDMatrix: sum of the PSD (PSDx + PSDy + PSDz)
% This sum is used to work with a single orientation
% independent spectrum
PSDMatrix(:,Nsignals+1) = sum(PSDMatrix,2);

% Get maximum PSD value in the range [freqmin freqmax]
[CoeffPeak,IndPeak] = max(PSDMatrix(FreqVect>=freqmin,Nsignals+1));

% Add the minimum frequency to obtain the correct dominant frequency
FreqPeak = FreqVect(IndPeak) + freqmin;

% Inspection Spectrum; 
% figure; plot(FreqVect,PSDMatrix);
% xlabel('Freq (Hz)'); ylabel('PSD');
% set(gca,'fontsize',20,'fontweight','bold','LineWidth',2);
% grid on; legend('PSDx','PSDy','PSDz','PSDsum');