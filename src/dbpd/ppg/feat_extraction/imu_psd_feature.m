function ratio_imu = imu_psd_feature(imu_segment, ppg_segment, fs_imu, fs_ppg)
    
    % Initalize parameters for psd feature IMU
    pwelchwin_imu = 3*fs_imu;       % Using pwelch for 3 seconds
    pwelchwin_ppg = 3*fs_ppg;
    perc_overlap = 0.5;
    noverlap_imu = perc_overlap*pwelchwin_imu;   % overlap between windows is 50 %
    noverlap_ppg = perc_overlap*pwelchwin_ppg;

    f_bin_res = 0.05;   % the treshold is set based on this binning --> so range of 0.1 Hz for calculating the PSD feature 
    nfft_ppg = 0:f_bin_res:fs_ppg/2;        % create equal binning for ppg and imu
    nfft_imu = 0:f_bin_res:fs_imu/2;

    [pxx1,f1] = pwelch(imu_segment,hann(pwelchwin_imu), noverlap_imu, nfft_imu, fs_imu);    % calculate psd using pwelch
    PSD_imu = sum(pxx1,2);     % sum over the three axis
    [pxx2,f2] = pwelch(ppg_segment,hann(pwelchwin_ppg), noverlap_ppg, nfft_ppg, fs_ppg);
    PSD_ppg = sum(pxx2,2);      % not relevant ...

    [~, max_PPG_psd_idx] = max(PSD_ppg);
    max_PPG_freq_psd = f2(max_PPG_psd_idx);

    %%---check dominant frequency (df) indices----%%
    [~, corr_imu_psd_df_idx] = min(abs(max_PPG_freq_psd-f1)); 

    df_idx = corr_imu_psd_df_idx-1:corr_imu_psd_df_idx+1;

    %%---check first harmonic (fh) frequency indices----%%
    [~, corr_imu_psd_fh_idx] = min(abs(max_PPG_freq_psd*2-f1)); 
    fh_idx = corr_imu_psd_fh_idx-1:corr_imu_psd_fh_idx+1;
    
    %%---check half dominant frequency----%%  Sometimes this is the dominant frequency and the first harmonic is related to the PPG!
    [~, corr_imu_psd_fdom_idx] = min(abs(max_PPG_freq_psd/2-f1)); 
    fdom_idx = corr_imu_psd_fdom_idx-1:corr_imu_psd_fdom_idx+1;

    acc_power_PPG_range = trapz(f1(df_idx), PSD_imu(df_idx)) + trapz(f1(fh_idx), PSD_imu(fh_idx)) + trapz(f1(fdom_idx), PSD_imu(fdom_idx));   % Add the correct indices and calculate power using trapezoidal integration
    acc_power_total = trapz(f1, PSD_imu);

    ratio_imu = acc_power_PPG_range/acc_power_total;        % calculate the relative power between power in IMU using the PPG range and total power in IMU

end
