%-------------------------------------------------------------------------------
% dec_tfd: Generate decimated (or sub-sampled) time-frequency distributions
%
% Syntax: tf=dec_tfd(x,kern_type,kern_params,Ntime,Nfreq,time_dec,freq_dec)
%
% Inputs: 
%      x = input signal (either real-valued signal of length-N or
%          complex-valued analytic signal of length-2N)
%
%      kern_type   = kernel type, either nonseparable, separable, Doppler-independent
%                    (DI) or lag-independent (LI):   { 'nonsep' | 'sep' | 'DI' | 'LI' }
%
%      kern_params = kernel parameters; different depending on kernel type
%
%                    if nonseparable kernel, then form: { kern_name, kern_param}
%                      e.g. tf=full_tfd( x, 'nonsep', { 'cw', 100 } );
%
%                    if Doppler-independent kernel, then of the form: 
%                      { window_length, window_type, [window_parameter (optional)] }
%                      e.g. tf=full_tfd( x, 'DI', {101,'hann'} );
%
%                    if lag-independent kernel, then of the form: 
%                      { window_length, window_type, [window_parameter (optional)] }
%                      e.g. tf=full_tfd( x, 'LI', {101,'cosh',0.01} );
%
%                    if separable kernel, then of the form: { doppler_window, lag_window }
%                    where doppler_window is of the form: { window_length, window_type,
%                    [window_parameter (optional)] }; same format for lag window
%                      e.g. tf=full_tfd( x, 'sep', { {101, 'cosh', 0.05}, {101,'hann'} } );
%
%     Ntime       = over-sampling in the time direction; only applicable for separable
%                   and lag-independent kernels
%
%     Nfreq       = over-sampling in the frequency direction; only applicable for separable
%                   and Doppler-independent kernels
%
%     time_dec  = decimation factor in the time domain; for the Doppler-independent kernel
%     this value can be either a vector or a scalar; for all other kernel types this
%     value is an integer scalar a.
%
%     freq_dec  = decimation factor in the frequency domain; for the lag-independent kernel
%     this value can be either a vector or a scalar; for all other kernel types this
%     value is an integer scalar b.
%
% Outputs: 
%     tf - time-frequency distribution of size (a/N) x (b/N) (non-separable kernel),
%          or (a/Ntime) x (b/Nfreq) (separable kernel),
%          or U x (b/Nfreq) (Doppler-independent kernel, U is length of time_dec vector),
%          or (a/Ntime) x V (lag-independent kernel, V is length of freq_dec vector)
%
% See also: DEC_NONSEP_GDTFD, DEC_SEP_GDTFD, DEC_LI_GDTFD, DEC_DI_GDTFD
%
% Example:
%     N=1024; Ntime=64; Nfreq=128;                                              
%     a=2; b=2;                                                       
%     ni=[100:2:900]; ki=[150:2:850];                                 
%                                                                     
%     x=gen_LFM(N,0.1,0.3)+gen_LFM(N,0.4,0.1);                        
%                                                                     
%     % non-separable kernel:                                         
%     c=dec_tfd(x,'nonsep',{'cw',100},N,N,a*4,b*4);                       
%     figure(1); clf; vtfd(c,x);                                      
%                                                                     
%     % separable kernel:                                             
%     c=dec_tfd(x,'sep',{{51,'hann'},{101,'hann'}},Ntime,Nfreq,a,b);    
%     figure(2); clf; vtfd(c,x);                                      
%                                                                     
%     % Doppler-independent kernel:                                   
%     c=dec_tfd(x,'DI',{101,'hann'},N,Nfreq,ni,b);                      
%     figure(3); clf; vtfd(c,x,1,ni);                                 
%                                                                     
%     % lag-independent kernel:                                       
%     c=dec_tfd(x,'LI',{51,'hann'},Ntime,N,a,ki);                       
%     figure(4); clf; vtfd(c,x,1,[],ki./(N*2));                       
      
% John M. O' Toole, University College Cork
% Started: 23-07-2014
%
% last update: Time-stamp: <2019-06-05 17:05:19 (otoolej)>
%-------------------------------------------------------------------------------
function tf=dec_tfd(x,kern_type,kern_params,Ntime,Nfreq,time_dec,freq_dec)
if(nargin<2 || isempty(kern_type)), kern_type='sep'; end
if(nargin<3 || isempty(kern_params)), kern_params={{51,'hann'},{101,'hann'}}; end
if(nargin<4 || isempty(time_dec)), time_dec=[]; end
if(nargin<5 || isempty(freq_dec)), freq_dec=[]; end
if(nargin<6 || isempty(Ntime)), Ntime=[]; end
if(nargin<7 || isempty(Nfreq)), Nfreq=[]; end


% set to 1 if want to see how much memory is used:
DBmem=0;


kern_type=lower(kern_type);
switch kern_type
  case { 'nonsep', 'ns', 'nonseparable', 'non-separable', 'non-sep' }
    %---------------------------------------------------------------------
    % 1. Non-separable kernel; Doppler-lag form: g[l,m]
    %---------------------------------------------------------------------
    nonsep_params=[];
    if(~iscell(kern_params))
        nonsep_name=kern_params;
    else
        nonsep_name=kern_params{1};
        if(length(kern_params)>1)
            nonsep_params=kern_params{2};
        end
    end
            
    tf=dec_nonsep_gdtfd(x,nonsep_name,nonsep_params,time_dec,freq_dec);

    
  case  { 'sep', 'separable' }
    %---------------------------------------------------------------------
    % 2. separable kernel; Doppler-lag form: G1[l]g2[m]
    %---------------------------------------------------------------------
    if(~iscell(kern_params) | (iscell(kern_params) & length(kern_params)<2) )
        error(['separable kernel parameters should be of the form: ' ...
               '{ {dopp_win_length,dopp_win_name}, {lag_win_length,lag_win_name} }']);
    end
    
    tf=dec_sep_gdtfd(x,kern_params{1},kern_params{2},time_dec,freq_dec,Ntime,Nfreq);
    
    
  case { 'di', 'doppler-independent', 'dopp.-indep', 'dopp-indep', 'pwvd', 'p-wvd' }
    %---------------------------------------------------------------------
    % 3. Doppler-independent kernel: Doppler-lag form: g2[m]
    %---------------------------------------------------------------------
    if(~iscell(kern_params))
        error(['Doppler-independent kernel parameters should be of the form: ' ...
               '{win_length,win_name}']);
    end
    
    tf=dec_di_gdtfd(x,kern_params,time_dec,freq_dec,Nfreq);

    
  case { 'li', 'lag-independent',  'lag-indep', 'swvd', 's-wvd' }
    %---------------------------------------------------------------------
    % 4. Lag-independent kernel: Doppler-lag form: G1[l]
    %---------------------------------------------------------------------
    if(~iscell(kern_params))
        error(['lag-independent kernel parameters should be of the form: ' ...
               '{win_length,win_name}']);
    end

    tf=dec_li_gdtfd(x,kern_params,time_dec,freq_dec,Ntime);
    
  otherwise
    warning('kern_type should be either: nonsep, sep, DI, or LI');
    tf=[];
end


if(DBmem), s=whos; fprintf('total memory used: mem=%s\n',disp_bytes(sum([s.bytes]))); end
