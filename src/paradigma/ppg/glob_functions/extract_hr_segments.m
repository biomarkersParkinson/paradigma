function [start_idx, end_idx] = extract_hr_segments(label_arr, threshold)
% Function which calculates the switches between high and low quality. It
% calculates the length of the high quality parts and returns the correct
% data indices for every high quality part longer than a specific threshold!
% Input:
%   - labelArray: Label array containin of 0 (low-quality) and 1 (high-quality)
%   - threshold: minimal required length for HR analysis (f.e. 30s = 30*fs=
%   900 samples

    start_idx = [];
    end_idx = [];
    
    label_arr = [0; label_arr; 0];              % padding to find switches if the label starts or ends with high quality label!

    % Find switches from 0 to 1
    zero_one_switch = find(diff(label_arr) == 1);
    
    % Find switches from 1 to 0
    one_zero_switch = find(diff(label_arr) == -1);
    
    % Ensure the lengths are the same
    if length(zero_one_switch) ~= length(one_zero_switch)
        error('Invalid label array');
    end
    
    % Calculate switch lengths for label 1
    switch_lengths = one_zero_switch - zero_one_switch;
    
    % Find switch lengths greater than the threshold
    long_switches = find(switch_lengths >= threshold);
    
    % Assign start and end indices for long switches
    for i = 1:length(long_switches)
        start_idx(i) = zero_one_switch(long_switches(i));
        end_idx(i) = one_zero_switch(long_switches(i))-1;
    end
end