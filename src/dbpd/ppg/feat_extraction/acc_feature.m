function acc_power_ratio = acc_feature(f1, PSD_acc, f2, PSD_ppg)
    % This function calculates the power ratio of the accelerometer signal in the PPG frequency range.
    % The power ratio is defined as the ratio of the power in the PPG frequency range to the total power.
    [~, max_PPG_psd_idx] = max(PSD_ppg);
    max_PPG_freq_psd = f2(max_PPG_psd_idx);
    
    %%---check dominant frequency (df) indices----%%
    [~, corr_acc_psd_df_idx] = min(abs(max_PPG_freq_psd-f1)); 
    
    df_idx = corr_acc_psd_df_idx-1:corr_acc_psd_df_idx+1;
    
    %%---check first harmonic (fh) frequency indices----%%
    [~, corr_acc_psd_fh_idx] = min(abs(max_PPG_freq_psd*2-f1)); 
    fh_idx = corr_acc_psd_fh_idx-1:corr_acc_psd_fh_idx+1;
    
    %%---calculate power ratio---%%
    acc_power_PPG_range = trapz(f1(df_idx), PSD_acc(df_idx)) + trapz(f1(fh_idx), PSD_acc(fh_idx));
    acc_power_total = trapz(f1, PSD_acc);
    
    acc_power_ratio = acc_power_PPG_range/acc_power_total;