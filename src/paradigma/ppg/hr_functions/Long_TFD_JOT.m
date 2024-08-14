%% Implementation using the toolbox of J. O'Toole
 
% Only the essential code is used. The function nonsep_gdtfd is code from
% the toolbox of J. O'Toole: Copyright © 2014, John M. O  Toole, University College Cork. All rights reserved.
function HR_smooth_tfd = Long_TFD_JOT(rel_ppg_tfd, MA, fs, kern_type, kern_params)

tfd = nonsep_gdtfd(rel_ppg_tfd, kern_type, kern_params);   % for now the wigner ville distribution but one could also use the smoothed pseudo WVD --> returns matrix of size NxN 

if MA.value == 1
    input = tfd';            
    tfd = filtfilt(MA.FC,1,input);
    tfd = tfd.';
end

%%----- Get time and frequency axis for tfd-----%%
[~,M]=size(tfd);
Ntime=size(tfd,1);   
    
ntime=1:Ntime; ntime=ntime./fs-1/fs;       % time array
Mh=ceil(M);
k=linspace(0,0.5,Mh);
k=k.*fs;                                   % frequency array

%%---Estimate HR using same approach as for given tfd----%%
[~, k_idx] = max(tfd, [], 1);


count = 0;

for i = 2:2:length(rel_ppg_tfd)/fs-4            % starting at 2 and ending at length -4 to discard the first and last 2 sec of the WVD which are influenced by boundary effects
    count = count + 1;
    rel_wvd_idx = ntime>=i & ntime<i+2;
    HR_smooth_tfd(count,1) = 60*mean(k(k_idx(rel_wvd_idx))); 

end

end