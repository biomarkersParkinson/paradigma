%-------------------------------------------------------------------------------
% check_dec_params_seq: Check and return arbitrary set from {k_1,k_2,...,k_J} samples
% could be for n (time) sample points also.
%
% Syntax: k=check_dec_params_seq(dec_param,N,time_freq_str)
%
% Inputs: 
%     dec_param,N,time_freq_str - 
%
% Outputs: 
%     k - 
%
% Example:
%     
%

% John M. O' Toole, University College Cork
% Started: 23-04-2014
%
% last update: Time-stamp: <2014-04-23 15:36:55 (otoolej)>
%-------------------------------------------------------------------------------
function [k,Nfreq,dec_param]=check_dec_params_seq(dec_param,N,time_freq_str,check_param)
if(nargin<3 || isempty(time_freq_str)), time_freq_str='Frequency'; end
if(nargin<4 || isempty(check_param)), check_param=1; end



%---------------------------------------------------------------------
% 0. check if parameters are ok:
%---------------------------------------------------------------------
dec_param=unique(dec_param);

if(~all(dec_param==fix(dec_param)))
  error([time_freq_str ' decimation parameters must be integers.']);
end    
if(any(dec_param>=N) || any(dec_param<0))
    error([time_freq_str ' decimation values need to within 0<=k<=N-1']);
end    
    

%---------------------------------------------------------------------
% 1. set parameters:
%---------------------------------------------------------------------
if(length(dec_param)==1)
    % decimation parameter is not a sequence:
    if(check_param)
        dec_param=check_dec_value(dec_param,N,time_freq_str);    
    end
    k=0:dec_param:N-1;
    
elseif(length(dec_param)>1)  
    k=dec_param;
else
    k=0:N-1;
end


if(any(k>=N) || any(k<0))
    error([time_freq_str ' decimation values need to within 0<=k<=N-1']);
end
Nfreq=length(k);


function dec_param=check_dec_value(dec_param,N,time_freq_str)
%---------------------------------------------------------------------
% check (or set) that  N/a is an integer
%---------------------------------------------------------------------
if(rem(N,dec_param)) 
  warning(['N not divisable by ' time_freq_str ' decimation factor. Increasing....']);
  while( rem(N,dec_param) )  
    dec_param=dec_param+1;   
  end
  disp( ['... decimation factor now=' num2str(dec_param)]); 
  if(dec_param>=N)
      error([time_freq_str ' decimation factor is equal to N. N is prime? Pick smaller ' ...
                          'decimation parameter']);
      dec_param=1;
  end
end
