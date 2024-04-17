function data_prob = sample_prob_final(ppg_prob, imu_label, fs)
% K.I. Veldkamp, PhD student AI4P, 29-02-24

%%--Assign probability to every individual data point!--%%
% Inputs:
%   - ppg_prob
%   - imu label
% Output:
% - data_prob: containing of an array with for every sample a probability.
% This can be mapped to the samples of the synced PPG-IMU

if ~exist('imu_label', 'var')
    imu_label = ones(length(ppg_prob));     % Function also works without having imu_label --> then based on ppg_prob alone!
end

epoch_length = 6; % in seconds
overlap = 5; % in seconds

% Number of samples in epoch
samples_per_epoch = epoch_length * fs;

% Calculate number of samples to shift for each epoch
samples_shift = (epoch_length - overlap) * fs;
n_samples = (length(ppg_prob) + overlap) * fs;
data_prob = zeros(n_samples,1);

prob_array = ppg_prob;
imu_array = imu_label;

for i = 1:n_samples
    start_idx = ceil((i-(samples_per_epoch-samples_shift))/fs);  %start_idx for the non starting and ending epochs is equal to ceil((data idx - n_overlap)/fs)
    end_idx = ceil(i/fs);
   
    %%-----Correct for first and last 6s epochs (those have less than 6 epochs to calculate labels and prob)-----%%
    if start_idx < 1 
        start_idx = 1;     % The first 5 epochs indices start at 1
    elseif end_idx>length(prob_array) 
        end_idx = length(prob_array); % The last 5 epochs indices end at the last label
    end

    prob = prob_array(start_idx:end_idx);
    label_imu = imu_array(start_idx:end_idx);
    data_prob(i) = mean(prob);
    data_label_imu(i) = int8(mean(label_imu) >= 0.5);       % Perform majority voting
end

data_prob(data_label_imu==0) = 0;    % Set prob to zero if majority voting of IMU is 0

end