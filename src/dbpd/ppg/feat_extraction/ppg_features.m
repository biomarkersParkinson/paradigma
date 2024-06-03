function [FeaturesPPG] = ppg_features(PPG,fs)
    % extract features from the PPG signal, per 6sec window (PPG input is a 6sec window of PPG signal)
    N_feat = 10;
    FeaturesPPG = zeros(1, N_feat);
    % Time-domain features
    absPPG = abs(PPG);
    FeaturesPPG(1) = var(PPG); % Feature 1: variance
    FeaturesPPG(2) = mean(absPPG); % Feature 2: mean
    FeaturesPPG(3) = median(absPPG); % Feature 3: median
    FeaturesPPG(4) = kurtosis(PPG); % Feature 4: kurtosis
    FeaturesPPG(5) = skewness(PPG); % Feature 5: skewness
    
    window = 3*fs;   % 90 samples for Welch's method => fr = 2/3 = 0.67 Hz --> not an issue with a clear distinct frequency
    overlap = 0.5*window; % 45 samples overlap for Welch's Method
    
    [P, f] = pwelch(PPG, window, overlap, [], fs);
    
    % Find the dominant frequency
    [~, maxIndex] = max(P);
    FeaturesPPG(6) = f(maxIndex);    % Feature 6: dominant frequency
    
    ph_idx = find(f >= 0.75 & f <= 3); % find indices of f in relevant physiological heart range 45-180 bpm
    [~, maxIndex_ph] = max(P(ph_idx)); %  Index of dominant frequency
    dominantFrequency_ph = f(ph_idx(maxIndex_ph)); % Extract dominant frequency 
    f_dom_band = find(f >= dominantFrequency_ph - 0.2 & f <= dominantFrequency_ph + 0.2); %
    FeaturesPPG(7) = trapz(P(f_dom_band))/trapz(P);  % Feature 7 = relative power
    
    
    % Normalize the power spectrum
    pxx_norm = P / sum(P);
    
    % Compute spectral entropy
    FeaturesPPG(8) = -sum(pxx_norm .* log2(pxx_norm))/log2(length(PPG)); % Feature 8 = spectral entropy --> normalize between 0 and 1! Or should we perform this operation at the min-max normalization! No because the values can come from different lengths!
    
    % Signal to noise ratio
    Signal = var(PPG);
    Noise = var(absPPG);
    FeaturesPPG(9) = Signal/Noise; % Feature 9 = surrogate of signal to noise ratio
    
    %% Autocorrelation features
    
    [acf, ~] = autocorr(PPG, 'NumLags', fs*3); % Compute the autocorrelation of the PPG signal with a maximum lag of 3 seconds (or 3 time the sampling rate)
    [peakValues, ~] = peakdet(acf, 0.01);
    sortedValues = sort(peakValues(:,2), 'descend');     % sort the peaks found in the corellogram
    if length(sortedValues) > 1
        FeaturesPPG(10) = sortedValues(2); % determine the second peak as the highest peak after the peak at lag=0, the idea is to determine the periodicity of the signal
    else
        FeaturesPPG(10) = 0;              % Set at 0 if there is no clear second peak
    end
    
    
    
    