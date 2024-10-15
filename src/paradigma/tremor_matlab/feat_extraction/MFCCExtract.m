function [MFCCFeatureOut] = MFCCExtract(Conf,Data)
% Input ###################################################################
% Conf - Structure
%     .Fs            - scalar - double - sampling frequency in Hz
%     .numFilters    - scalar - double -  number of filters in the mel scale
%     .NumMelCoeff   - scalar - double -  number of MFCCs coefficients
%     .StrAxis       - string - specifying axis (or sum) to be considered for evaluating the spectrogram
%     .MaxFreqFilter - scalar - double - maximal frequency (Hz) to be considred in the bank filter
%     .MFCCwin       - scalar - double - subwindow in seconds adopted for evaluating the spectrogram
%
% Data - matrix (Nsamples x Nsignals) - double - columns: IMU signals ([x y and z])
%
%
% Output
% MFCCFeatureOut - matrix - double - .'TypeOfSignal'MFCC'coefficient number'
% #########################################################################

[Nsamples,Nsignals] = size(Data); % get number of samples and number of signals

Fs          = Conf.Fs;          % sampling rate
numFilters  = Conf.numFilters;  % number of filters
NumMelCoeff = Conf.NumMelCoeff; % number of MFCCs coefficients
StrAxis     = Conf.StrAxis;     % axis or sum of axes to be considered in the spectrogram

filterbankStart = 0;                  % frequency for starting the bank filter
filterbankEnd   = Conf.MaxFreqFilter; % maximum frequency in the bank filter

numBandEdges = numFilters + 2;         % number of filter edges

NFFT         = Conf.MFCCwin*Fs;        % number of samples for FFT = number of samples of the adopted subwindow to evaluate the spectrogram

fresol       = Fs/NFFT;                % frequency resolution if 0.5 Hz

filterBank   = zeros(numFilters,NFFT); % initializing the bank filter
                
x = filterbankStart:0.01:filterbankEnd; % linear spacing in the bank filter frequency range

melscale = 64.875*log10( 1 + x./17.5); % creating mel scale

xlinmel = linspace(melscale(1),melscale(end),numBandEdges); % equally intervals in the mel scale

bandEdges = 17.5*(10.^(xlinmel/64.875) - 1);    % getting the edges in the frequency domain

bandEdgesBins = round((bandEdges/Fs)*NFFT) + 1; % rouding the edges

% Getting each filter in the frequency domain
for ii = 1:numFilters
    filt = triang(bandEdgesBins(ii+2)-bandEdgesBins(ii)); % Triangular filters based on the edges
    leftPad = bandEdgesBins(ii);             % introduce zero padding on the left
    rightPad = NFFT - numel(filt) - leftPad; % introduce zero padding on the right
    filterBank(ii,:) = [zeros(1,leftPad),filt',zeros(1,rightPad)]; % concatenate the vector
    filterBank(ii,:) = filterBank(ii,:)/(fresol*sum(filterBank(ii,:))); % Normalize the filter coefficients by the bandpass area
end

% Check the filter profiles if needed
% frequencyVector = (Fs/NFFT)*(0:NFFT-1);
% plot(frequencyVector,filterBank');
% xlabel('Hz')
% axis([0 frequencyVector(NFFT/2) 0 1])

overlap = round(0.8*NFFT); % ovelap between each subwindow for evaluating the spectrogram

% Extracting the MFCC according to axis (or sum) specified when evaluating
% the spectrogram
switch StrAxis
    case 'X'
        Signal = Data(:,1); % Get the signal
        % Evaluate the spectrogram
        [S1,f,t] = stft(Signal,Fs,"Window",hann(NFFT,'periodic'),...
            'OverlapLength',overlap,"FrequencyRange","twosided");
        S = abs(S1); % Get the absolute value
    case 'Y'
        Signal = Data(:,2);
        [S2,f,t] = stft(Signal,Fs,"Window",hann(NFFT,'periodic'),...
            'OverlapLength',overlap,"FrequencyRange","twosided");
        S = abs(S2);
    case 'Z'
        Signal = Data(:,3);
        [S3,f,t] = stft(Signal,Fs,"Window",hann(NFFT,'periodic'),...
            'OverlapLength',overlap,"FrequencyRange","twosided");
        S = abs(S3);
    case 'Sum'
        [S1,f,t] = stft(Data(:,1),Fs,"Window",hann(NFFT,'periodic'),...
            'OverlapLength',overlap,"FrequencyRange","twosided");
        [S2,f,t] = stft(Data(:,2),Fs,"Window",hann(NFFT,'periodic'),...
            'OverlapLength',overlap,"FrequencyRange","twosided");
        [S3,f,t] = stft(Data(:,3),Fs,"Window",hann(NFFT,'periodic'),...
            'OverlapLength',overlap,"FrequencyRange","twosided");
        S = abs(S1) + abs(S2) + abs(S3); % Get the sum of the absolute values across axes 
end

% Applying the filter bank
Spec      = filterBank*S;

% Obtaning the cepstral coefficients
CepsCoeff = cepstralCoefficients(Spec,'NumCoeffs',NumMelCoeff);

Ceps = mean(CepsCoeff); % Taking the average of the cepstral coefficients across time

MFCCFeatureOut = Ceps; % Defining the output
