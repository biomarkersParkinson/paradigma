% #########################################################################
% Input
% Conf - Structure
%     .WindowSizeTime - scalar - double - time of the window in seconds
%     .Fs             - scalar - double - sampling frequency in Hz
%
% Data_In - cell      - array - double - Nsamples x Nchannels - IMU signal in
%                                  columns [x y z]
%
% t - cell            - array - double - Nsamples x 1 - IMU time stamp - 
%                                  Unix time corrected (interpolated) 
%
% Type of the signal  - string - IMU type of signal 'Gy' - gyroscope
%                                                   'Ac' - acceleration
%
% #########################################################################
% Output
% Features - structure
%         .WindowIniTime  - array - double - Nwindows x 1 - Unix Corrected Time (after interpolation) -
%                                                           Initial time stamp of the window
%
%         .Derivatives   - table - double - Nwindows x 3 - Mean angular velocity absolute derivatives for each axis (columns):
%                                             .GyMeanDx - x-axis
%                                             .GyMeanDy - y-axis
%                                             .GyMeanDz - z-axis
%
%         .PowerAxis       - table - double - Nwindows x 3 - Full band power axis (0.5-25 Hz) (columns):
%                                             .GyPowerX - x-axis
%                                             .GyPowerY - y-axis   
%                                             .GyPowerZ - z-axis
%
%         .DomTremorPowerAxis       - table - double - Nwindows x 3 - Tremor dominant power (3-7 Hz) for each axis (columns):
%                                             .GyLTreDomPowerX - x-axis
%                                             .GyLTreDomPowerY - y-axis   
%                                             .GyLTreDomPowerZ - z-axis
%
%         .FreqPeak      - table - double - Nwindows x 1 - Dominant
%         frequency in the PSD summed spectrum (PSD = PSDx + PSDy + PSDz) within the range 1-25Hz per
%         window
%                                             .GyFreqPeak - frequency of
%                                             the PSD peak

%         .PSDTremorFeatures  - table - double - Nwindows x 3 - PSD-based
%         featured extracted in the tremor range (3-7) Hz per window 
%         - orientation invariant (PSD = PSDx + PSDy + PSDz)
%                                             .GyLTreBandPower  - power in the tremor range
%                                             .GyLTreFreqPeak   - dominant frequency in the tremor range
%                                             .GyLTreDomPower   - dominant power 1.25 Hz around the tremor peak   
%
%         .PSDHighTremorFeatures  - table - double - Nwindows x 3 - PSD-based
%         featured extracted in the high tremor range (7-12) Hz per window
%         - orientation invariant (PSD = PSDx + PSDy + PSDz)
%                                             .GyHTreBandPower - power in the high tremor range
%                                             .GyHTreFreqPeak  - dominant frequency in the high tremor range
%                                             .GyHTreDomPower  - dominant power 1.25 Hz around the high tremor peak   
%
%         .MelCepsCoeff  - table - double - Nwindows x 12 -
%         Mel-Frequency cepstral coefficients from 1 to 12 per window -
%         orientation invariant (sum spectrograms x, y, z)
%                                             .GyMFCC1   - 1st MFCC coefficient
%                                             .GyMFCC2   - 2nd  MFCC coefficient
%                                             ...
%                                             .GyMFCC12  - 12th MFCC coefficient
%
%
%         .PowerArmActv - table - double Nwindows x 1 - Power in the Arm Activity
%                                          frequency range (0.5 - 3 Hz)
%                                          .GyArmActvPower - power in the range (0.5 - 3 Hz)
%
%
%         .PowerHighFreq - table - double Nwindows x 1 - Power in the high frequency domain (12-25 Hz)
%                                          .GyHighFreqPower - power in the
%                                          range (12-25 Hz)
% 
%         .TremorProb - array - double - Nwindows x 1 - tremor
%         probability obtained by logistic regression based on MFCCs
%         features
% 
%         .TremorHat  - array - double - Nwindows x 1    - tremor label obtained after applying threshold in the tremor probability 
%         .RestTremorHat - array - double - Nwindows x 1 - rest tremor
%         label after logical AND operation between TremorHat and SpectralCheck (PSD peak falls within tremor domain [3 - 7 Hz])
% #########################################################################

function [Features] = TremorFeaturesAndClassification(Conf,Data_In,t,TypeOfSignal)

% Load Classifiers obtained from PD@Home
load('TremorClassifier2.mat'); % Load the updated tremor model
load('Threshold_total.mat');   % Load the threshold

TremorThr = Threshold_total;   % Threshold at 95% specificity

load('MeanVectorPDhome.mat');  % Scaling parameters
load('SigmaVectorPDhome.mat'); % Scaling parameters

% Getting parameters
Fs = Conf.Fs;                           % Sampling frequency
WindowSizeTime = Conf.WindowSizeTime;   % window interval in seconds

% Evaluating slicing parameters
DeltaSample = floor(WindowSizeTime*Fs);               % number of samples for skipping to get the next window
                                                      % Introduce overlap between samples if needed
NumWindows = floor(length(Data_In(:,1))/DeltaSample); % number of windows given the segment size (Data_In)

% Initialize variables
WindowIniTime      = [];
DomTremorPowerAxis = [];
PowerAxis          = [];

MelCepsCoeff  = [];
Derivatives   = [];
PSDTremorFeatures     = [];
PSDHighTremorFeatures = []; 
FreqPeak      = [];
ArmActvPower  = [];
HighFreqPower = [];

TremorProb    = [];
TremorHat     = [];
PeakFreqTremorCheck = [];
RestTremorHat = [];

% Loop across the windows
for kk = 1:NumWindows  

    % Get the inital time stamp
    WindowIniTime(kk)        = t((kk-1)*DeltaSample+1);

    % Slice data according WindowSizeTime
    sig_x  = Data_In(((kk-1)*DeltaSample+1):kk*DeltaSample,1);
    sig_y  = Data_In(((kk-1)*DeltaSample+1):kk*DeltaSample,2);
    sig_z  = Data_In(((kk-1)*DeltaSample+1):kk*DeltaSample,3);

    DataMatrix = [sig_x sig_y sig_z];                               % Concatenating Data - Signal Matrix           

    %% PSD estimation - Welch periodogram function parameters - Evaluate the Spectrum
    % Configuration for power spectrum density (PSD) estimation
    ConfPSD.Fs        = Fs;        % sampling frequency in Hz
    ConfPSD.pwelchwin = 3*Fs;      % number of samples for the subwindows used in the periodogram
    ConfPSD.noverlap  = round(0.8*ConfPSD.pwelchwin);  % number of samples for overlapping subwindows 
    ConfPSD.nfft      = 4*Fs;      % NFFT number of samples - keep spectral resolution in 0.25 Hz
    ConfPSD.freqmax   = 25;        % Maximum frequency to be considered
    ConfPSD.freqmin   = 1;         % Lower frequency limit used in PSDest
                                   % for estimating the dominant frequency in
                                   % the range [freqmin freqmax] Hz. This low
                                   % frequency cut is due to gyroscope drift and the presence of enhanced low frequency components in this case.
    
    [PSDMatrix,FreqVect,FreqPeak(kk)] = PSDEst(ConfPSD,DataMatrix); % Get the PSD Matrix
    
    % Check extra criterion for rest tremor: Does the dominant peak within
    %                                        1-25 Hz falls into the tremor range (3-7) Hz
    
    PeakFreqTremorCheck(kk) = (FreqPeak(kk) >= 3 & FreqPeak(kk) <= 7);

    %% Mel-Cepstrum Coefficients Feature Extraction
    % Configuration for MFCCs extraction
    ConfMFCC.Fs            = Fs;    % Sampling Frequency
    ConfMFCC.numFilters    = 15;    % Number of filters to be used in the Mel-scale
    ConfMFCC.NumMelCoeff   = 12;    % number of MFCCs coefficients
    ConfMFCC.StrAxis       = 'Sum'; % Evaluation based on a summed spectrogram, i.e., orientation independent
    ConfMFCC.MaxFreqFilter = 25;    % Maximal frequency in the bank filter
    ConfMFCC.MFCCwin       = 2;     % Subwindow in seconds for evaluating the spectrogram

    [MelCepsCoeffAux] = MFCCExtract(ConfMFCC,DataMatrix); % Perform MFCC extraction

    %% Tremor classification
    DataScaled = (MelCepsCoeffAux - MeanVector)./SigmaVector; % Scale MFCCs features for classification
    TremorProb(kk)  = predict(Mdl,DataScaled);                % Tremor probability based on logistic regression classifier
    
    TremorHat(kk)      = (TremorProb(kk) > TremorThr); % Boolean
    RestTremorHat(kk)  = (TremorProb(kk) > TremorThr) & (PeakFreqTremorCheck(kk)); % Boolean

    MelCepsCoeff = vertcat(MelCepsCoeff,MelCepsCoeffAux);    % Concatenate values
    clear('MelCepsCoeffAux'); % clear temp variable

    %% Tremor Severity and arm activity features (step 5 for eScience toolbox)

    % Derivatives Evaluation - Get the mean absolute derivative values
    DerivativesAux   = DerivativesExtract(sig_x,sig_y,sig_z,Fs); % Evaluate mean signals' absolute derivatives values
    Derivatives      = vertcat(Derivatives,DerivativesAux);         % Concatenate values - store Derivatives
    clear('DerivativesAux');                                        % Clear the auxiliar strucuture

    % Parameters for obtaning spectral features
    fSpectResol       = Fs/ConfPSD.nfft;                       % Spectral resolution: fixed Fs/(4*Fs) = 0.25 Hz
    ConfPSDExtr.SpectralIntervalPeak = round(0.5/fSpectResol); % The interval for calculating the dominant power
    % is 0.5 Hz for each side of the peak,
    % which implies in different number of
    % samples in the freq domain depending
    % on the spectral resolution

    ConfPSDExtr.StrAxis = 'Sum';         % PSD features will be evaluated in the PSDsum domain, i.e., PSDx+PSDy+PSDz: orientation independent
    ConfPSDExtr.nfft    = 4*Fs;          % Number of points FFT
    ConfPSDExtr.Fs      = Fs;            % Sampling frequency
    ConfPSDExtr.MaxFreqOverallBand = 25; % Maximal frequency for evaluating total power

    % Low Tremor Features
    ConfPSDExtr.freqrange = [3 7];       % IMPORTANT - define here the bandwidth for tremor band
    [PSDTremorFeaturesAux] = PSDExtrOpt(ConfPSDExtr,PSDMatrix,FreqVect);     % Get Low Freq Tremor Features
    PSDTremorFeatures      = vertcat(PSDTremorFeatures,PSDTremorFeaturesAux);   % Store low freq tremor features
    clear('PSDTremorFeaturesAux');                                              % Clear temp variable

    % In the tremor range: get PSD Dominant power for each axis
    
    ConfAxis.StrAxis = 'X';
    ConfAxis.freqrange = [3 7];       % Frequency range to get tremor dominant power
    ConfAxis.nfft    = 4*Fs;          % Number of points FFT
    ConfAxis.Fs      = Fs;            % Sampling frequency
    ConfAxis.SpectralIntervalPeak = round(0.5/fSpectResol); % The interval for calculating the dominant power
    ConfAxis.MinFreqOverallBand = 0.5; % Minimal frequency for evaluating total power
    ConfAxis.MaxFreqOverallBand = 25;  % Maximal frequency for evaluating total power
    
    [PSD_X]  = PSDExtrAxis(ConfAxis,PSDMatrix,FreqVect);
    
    ConfAxis.StrAxis = 'Y';
    [PSD_Y]  = PSDExtrAxis(ConfAxis,PSDMatrix,FreqVect);
    
    ConfAxis.StrAxis = 'Z';
    [PSD_Z]  = PSDExtrAxis(ConfAxis,PSDMatrix,FreqVect);

    % Get the fixed bandwith dominant power 1.25 Hz around the peak in the
    % tremor range
    
    PSDPowerAxisAux               = [PSD_X(1) PSD_Y(1) PSD_Z(1)];
    PSDTremorDomPowerAxisAux      = [PSD_X(2) PSD_Y(2) PSD_Z(2)];
    
    PowerAxis                  = vertcat(PowerAxis,PSDPowerAxisAux);
    DomTremorPowerAxis         = vertcat(DomTremorPowerAxis,PSDTremorDomPowerAxisAux);
    
    clear('PSD_X','PSD_Y','PSD_Z','PSDTremorDomPowerAxisAux','PSDPowerAxisAux');

    % Get features in the high tremor range
    ConfPSDExtr.freqrange = [7 12];                                              % IMPORTANT - define here the bandwidth 
    [PSDHighTremorFeaturesAux] = PSDExtrOpt(ConfPSDExtr,PSDMatrix,FreqVect);  % Get High Freq Tremor Features
    PSDHighTremorFeatures      = vertcat(PSDHighTremorFeatures,PSDHighTremorFeaturesAux);   % Store
    clear('PSDHighTremorFeaturesAux');                                                      % Clear temp variable

    % Get Power in the arm-activity
    ConfBP.freqrange = [0.5 3];    % Frequency range   
    ConfBP.StrAxis = 'Sum'; % Axis or sum
    ConfBP.nfft = 4*Fs;     % number of samples for FFT
    ConfBP.Fs = Fs;         % Sampling frequency
    
    [ArmActvPowerAux] = PSDBandPower(ConfBP,PSDMatrix,FreqVect);     % Get arm-activity bandpower
    ArmActvPower      = vertcat(ArmActvPower,ArmActvPowerAux);       % Store 
    clear('ArmActvPowerAux');                                        % Clear temp variable

    % Get Power in high frequency band 
    ConfBP.freqrange = [12 25];       % Frequency range
    [HighFreqPowerAux] = PSDBandPower(ConfBP,PSDMatrix,FreqVect); % Get high frequency bandpower
    HighFreqPower      = vertcat(HighFreqPower,HighFreqPowerAux); % Store 
    clear('HighFreqPowerAux');                                    % Clear temp variable
   
end
% Organizing Output as a table

%% Frequency Peak
VarName = {'GyFreqPeak'};
FreqPeak = array2table(FreqPeak','VariableNames',VarName);

%% MFCCs
% Creating MFCCs names to be used
for pp = 1:ConfMFCC.NumMelCoeff
    CepsName{1,pp} = [TypeOfSignal,'MFCC',num2str(pp)];
end

VarName = CepsName;
% Defining the output
MelCepsCoeff = array2table(MelCepsCoeff,'VariableNames',VarName);
clear('VarName');

%% Derivatives
VarName = [{[TypeOfSignal,'MeanDx']},{[TypeOfSignal,'MeanDy']},{[TypeOfSignal,'MeanDz']}];

% Converting array to table
Derivatives = array2table(Derivatives,"VariableNames",VarName);
clear('VarName');

%% PSD Features Arm Activity Band
VarName      = {[TypeOfSignal,'ArmActvPower']};
ArmActvPower = array2table(ArmActvPower,"VariableNames",VarName);
clear('VarName');

%% PSD Features High Frequency Band
VarName = {[TypeOfSignal,'HighFreqPower']};
HighFreqPower = array2table(HighFreqPower,"VariableNames",VarName);
clear('VarName');

% Tremor Band
StrBand   = 'LTre';                 % String specifying the bandwidth in 
                                    % the obtained features name

VarName = {[TypeOfSignal,StrBand,'BandPower'], ...
           [TypeOfSignal,StrBand,'FreqPeak'],...
           [TypeOfSignal,StrBand,'DomPower']};

PSDTremorFeatures = array2table(PSDTremorFeatures,'VariableNames',VarName);
clear('VarName');

%% Dominant tremor power in each axis
VarName = {[TypeOfSignal,StrBand,'DomPowerX'],...
           [TypeOfSignal,StrBand,'DomPowerY'],...
           [TypeOfSignal,StrBand,'DomPowerZ']};

DomTremorPowerAxis  = array2table(DomTremorPowerAxis ,'VariableNames',VarName);
clear('VarName');

%% Full band: (0.5-25 Hz) in each axis
VarName = {[TypeOfSignal,'PowerX'],...
           [TypeOfSignal,'PowerY'],...
           [TypeOfSignal,'PowerZ']};

PowerAxis  = array2table(PowerAxis ,'VariableNames',VarName);
clear('VarName');

%% Features in High Tremor Band (7 - 12 Hz)
StrBand   = 'HTre';      % String specifying the bandwidth in the obtained features name
VarName = {[TypeOfSignal,StrBand,'BandPower'], ...
           [TypeOfSignal,StrBand,'FreqPeak'],...
           [TypeOfSignal,StrBand,'DomPower']};

PSDHighTremorFeatures = array2table(PSDHighTremorFeatures,'VariableNames',VarName);
clear('VarName');

if NumWindows == 0 % If there are not enough samples for having at least one window, the output structure is empty
    
    Features = [];
else % otherwise, Features receive the concatenated arrays and tables evaluated for the windows

    Features.WindowIniTime = WindowIniTime';
    Features.Derivatives   = Derivatives;
    Features.PowerAxis     = PowerAxis; 
    Features.DomTremorPowerAxis  = DomTremorPowerAxis;
    
    Features.FreqPeak      = FreqPeak;
    Features.PSDTremorFeatures  = PSDTremorFeatures;
    Features.PSDHighTremorFeatures  = PSDHighTremorFeatures;
    
    Features.MelCepsCoeff  = MelCepsCoeff;
    Features.PowerArmActv       = ArmActvPower;
    Features.PowerHighFreq      = HighFreqPower;
    
    Features.TremorProb    = TremorProb';
    Features.TremorHat     = TremorHat';
    Features.RestTremorHat = RestTremorHat';

end
