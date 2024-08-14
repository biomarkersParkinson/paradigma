function HR_smooth_tfd = PPG_TFD_HR(rel_ppg_tfd, tfd_length, MA, fs, kern_type, kern_params)
    % Function to estimate the HR using a particular TFD method, such as WVD,
    % SPWVD or other kernel methods over segments of 30 s. If the remaining segment
    % length is longer than 30 s it is calculated over the entire length to
    % include all data into the analysis. 
    
    % Input:
    % - rel_ppg_wvd: ppg signal with an addition of 2s on both sides to
    % overcome boundary effects
    % - tfd_length: set the length to calculate time frequency distribution
    % - MA: struct containing the information to perform a moving average
    % filter over the TFD to overcome fundamental frequency switching
    % - fs: sample frequency
    % - kern_type: specified as a string such as 'wvd' for the wigner ville
    % distribution
    % - kern_params: kernel specifications, not relevant for 'wvd', but they
    % are for 'spwvd', 'swvd' etc.
    
    % Output:
    % - HR_smooth_tfd: array containing an estimation of the HR for every 2 s
    % of the relevant epoch (without the additional 2 s on both sides)
    
    if ~exist('kern_params', 'var')
        kern_params = {};     % Function also works without having imu_label --> then based on ppg_prob alone!
    end
    
    edge_add = 4;       % adding an additional 4 sec (2 s to both sides) to overcome boundary effects/discontinuities of the WVD, in SPWVD this effect is already diminished due to a double kernel function
    epoch_length = tfd_length + edge_add;
    segment_length = (length(rel_ppg_tfd)-edge_add*fs)/fs;              % substract the 4 added sec to obtain the original segment length
    
    if segment_length > epoch_length
        n_segments = floor(segment_length/tfd_length); %% Dit moet aangepast worden!!
    else 
        n_segments = 1;  % for HR segments which are shorter than 30 s due to uneven start and end of the segment which can make the segment 28 s if 1 s is substracted
    end
    
    ppg_segments = cell(n_segments,1);
    
    for i = 1:n_segments                        % Split segments in 30 s PPG epochs 
    
        if i ~= n_segments          
            ppg_segments{i} = rel_ppg_tfd(1 + (i-1)*tfd_length*fs: (i*tfd_length+edge_add)*fs);
    
        else 
            
            ppg_segments{i} = rel_ppg_tfd(1 + (i-1)*tfd_length*fs:end);
    
        end
    
    end
    
    HR_smooth_tfd = [];
    
    for j = 1:n_segments                    % Calculate the HR
        ppg = ppg_segments{j};
        HR_tfd = Long_TFD_JOT(ppg, MA, fs, kern_type, kern_params);
        HR_smooth_tfd = [HR_smooth_tfd; HR_tfd];
    end
    