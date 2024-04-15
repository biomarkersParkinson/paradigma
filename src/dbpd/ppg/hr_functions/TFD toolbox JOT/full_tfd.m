%-------------------------------------------------------------------------------
% full_tfd: 
%
% Syntax: tf=full_tfd(x,kern_type,kern_params,Ntime,Nfreq)
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
% Outputs: 
%     tf          =  time-frequency distribution of size N x N (nonseparable kernel), 
%                    or Ntime x N (lag-independent kernel), 
%                    or N x Nfreq (Doppler-independent kernel), 
%                    or Ntime x Nfreq (separable kernel)
%
% See also: NONSEP_GDTFD, SEP_GDTFD, LI_GDTFD, DI_GDTFD
%
% Examples:
%    N=512;
%    x=gen_LFM(N,0.1,0.3) + gen_LFM(N,0.4,0.04);
%    
%    %  nonseparable kernel (Choi-Williams kernel):
%    tf=full_tfd(x,'nonsep',{'cw',10});
%    figure(1); clf; vtfd(tf,x);
%    
%    %  separable kernel:
%    tf=full_tfd(x,'sep',{{51,'hann'},{101,'hann'}},256,256);
%    figure(2); clf; vtfd(tf,x);
%    
%    %  Doppler-independent kernel:
%    tf=full_tfd(x,'DI',{101,'hann'},256,[]);
%    figure(3); clf; vtfd(tf,x);
%    
%    %  lag-independent kernel:
%    tf=full_tfd(x,'LI',{51,'hann'},[],256);
%    figure(4); clf; vtfd(tf,x);



% John M. O' Toole, University College Cork
% Started: 11-06-2014
%
% last update: Time-stamp: <2017-05-30 10:00:46 (otoolej)>
%-------------------------------------------------------------------------------
function tf=full_tfd(x,kern_type,kern_params,Ntime,Nfreq)
if(nargin<2 || isempty(kern_type)), kern_type='sep'; end
if(nargin<3 || isempty(kern_params)), kern_params={{51,'hann'},{101,'hann'}}; end
if(nargin<4 || isempty(Ntime)), Ntime=[]; end
if(nargin<5 || isempty(Nfreq)), Nfreq=[]; end


% set to 1 if want to see how much memory is used:
DBmem=1;


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
            
    tf=nonsep_gdtfd(x,nonsep_name,nonsep_params);

    
  case  { 'sep', 'separable' }
    %---------------------------------------------------------------------
    % 2. separable kernel; Doppler-lag form: G1[l]g2[m]
    %---------------------------------------------------------------------
    if(~iscell(kern_params) | (iscell(kern_params) & length(kern_params)<2) )
        error(['separable kernel parameters should be of the form: ' ...
               '{ {dopp_win_length,dopp_win_name}, {lag_win_length,lag_win_name} }']);
    end
    
    tf=sep_gdtfd(x,kern_params{1},kern_params{2},Ntime,Nfreq);
    
    
  case { 'di', 'doppler-independent', 'dopp.-indep', 'dopp-indep', 'pwvd', 'p-wvd' }
    %---------------------------------------------------------------------
    % 3. Doppler-independent kernel: Doppler-lag form: g2[m]
    %---------------------------------------------------------------------
    if(~iscell(kern_params))
        error(['Doppler-independent kernel parameters should be of the form: ' ...
               '{win_length,win_name}']);
    end
    
    
    tf=di_gdtfd(x,kern_params,Nfreq);

    
  case { 'li', 'lag-independent',  'lag-indep', 'swvd', 's-wvd' }
    %---------------------------------------------------------------------
    % 4. Lag-independent kernel: Doppler-lag form: G1[l]
    %---------------------------------------------------------------------
    if(~iscell(kern_params))
        error(['lag-independent kernel parameters should be of the form: ' ...
               '{win_length,win_name}']);
    end

    tf=li_gdtfd(x,kern_params,Ntime);
    
  otherwise
    warning('kern_type should be either: nonsep, sep, DI, or LI');
    tf=[];
end


if(DBmem), s=whos; fprintf('total memory used: mem=%s\n',disp_bytes(sum([s.bytes]))); end
