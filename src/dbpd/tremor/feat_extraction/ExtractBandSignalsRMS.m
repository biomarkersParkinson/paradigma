function [RMSValuesOut] = ExtractBandSignalsRMS(Conf,Data,TremorHatLabel)

% Input ###################################################################
% Conf - Configuration Structure
%     .Fs - scalar - double - sampling frequency
%
% Data         - matrix (Nsamples x Nsignals) - double - columns: IMU signals ([x y and z])
%
% TremorHatLabel - scalar - double - rest tremor label
%
% Output
% RMSValuesOut - matrix - double - (1 x 7) .'TypeOfSignal'TremorRMSx   - root mean square value in the tremor range (3 - 7) Hz - x axis
%                                          .'TypeOfSignal'TremorRMSy   - root mean square value in the tremor range (3 - 7) Hz - y axis
%                                          .'TypeOfSignal'TremorRMSz   - root mean square value in the tremor range (3 - 7) Hz - z axis
%                                          .'TypeOfSignal'SumTremorRMS - root mean square value in the tremor range (3 - 7) Hz - sum: (RMSx + RMSy + RMSz)
%                                          .'TypeOfSignal'SumGaitRMS   - root mean square value in the gait range (0.4 - 2) Hz - sum: (RMSx + RMSy + RMSz)
%                                          .'TypeOfSignal'SumHighTremorRMS - root mean square value in the high tremor range (7 - 12) Hz - sum: (RMSx + RMSy + RMSz)
%                                          .'TypeOfSignal'VeryHighBandRMS  - root mean square value in the high tremor range (12- 20) Hz  - sum: (RMSx + RMSy + RMSz)
% #########################################################################

if TremorHatLabel == 1

    Fs = Conf.Fs; % Sampling frequency

    % Load filter coefficients previously designed and saved in matlab
    % 6th order bandpass Butterworth filters

    switch Fs % Check the sampling frequency and load the corresponding filter
        case 100
            load('BP_Butter_filter_0_4_2_fs_100Hz.mat');
            load('BP_Butter_filter_3_7_fs_100Hz.mat');
            load('BP_Butter_filter_7_12_fs_100Hz.mat');
            load('BP_Butter_filter_12_20_fs_100Hz.mat');

        case 50

            load('BP_Butter_filter_0_4_2_fs_50Hz.mat');
            load('BP_Butter_filter_3_7_fs_50Hz.mat');
            load('BP_Butter_filter_7_12_fs_50Hz.mat');
            load('BP_Butter_filter_12_20_fs_50Hz.mat');
    end
    %% Filtering on the required ranges
    FilGaitBand       = filtfilt(SOSGait,GGait,Data);
    FilTremorBand     = filtfilt(SOSTre,GTre,Data);
    FilHighTremorBand = filtfilt(SOSHTre,GHTre,Data);
    FilVeryHighBand   = filtfilt(SOSVHTre,GVHTre,Data);

    %% Evaluating RMS value
    GaitRMS         = rms(FilGaitBand); % returns a vector [RMSx RMSy RMSz]
    TremorRMS       = rms(FilTremorBand);
    HighTremorRMS   = rms(FilHighTremorBand);
    VeryHighBandRMS = rms(FilVeryHighBand);

    % Sum across axes
    SumGaitRMS         = sum(GaitRMS);
    SumTremorRMS       = sum(TremorRMS);
    SumHighTremorRMS   = sum(HighTremorRMS);
    SumVeryHighBandRMS = sum(VeryHighBandRMS);

    % Getting the RMS in the tremor range for each axis to be included in the output
    TremorRMSx = TremorRMS(1);
    TremorRMSy = TremorRMS(2);
    TremorRMSz = TremorRMS(3);

    % Concatenate the features
    RMSValuesOut = [TremorRMSx  TremorRMSy TremorRMSz SumTremorRMS SumGaitRMS SumHighTremorRMS SumVeryHighBandRMS];

else

    RMSValuesOut = NaN(1,7);

end