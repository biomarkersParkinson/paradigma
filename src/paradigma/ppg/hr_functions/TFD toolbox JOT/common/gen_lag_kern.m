%-------------------------------------------------------------------------------
% gen_lag_kern: smoothing window for Lag-kernel g2[m]; provides some checking of 
%               the parameters
%
% Syntax: [g2,P,Ph_floor,Nfreq]=gen_lag_kern(win_params,N,Nfreq)
%
%
%  Input:
%       win_params - cell of {win_length,win_type,[win_param],[lag_or_freq]}
%                     e.g. {11,'hamm',0,1}
%                     the parameter lag_or_freq is either:
%                       0 = to define window in the lag-domain
%                       1 = to define window in the frequency-domain
%       N          - lenght of signal
%       Nfreq      - number of sample points in the frequency direction
%
%  Output:
%       g2       - lag kernel g_2(2mT)
%       P        - length of g2 (as may have changed)
%       Ph_floor - floor(P/2)
%       Nfreq    - Nfreq (as may have changed)
%
%  See also: GEN_DOPPLER_KERN, GEN_DOPPLER_LAG_KERN, GET_WINDOW, PADWIN
%
%  Example:
%       g2=get_lag_kernel({61,'hann'},128,512);
%

% John M. O' Toole, University College Cork
% Started: 16-04-2014
%
% last update: Time-stamp: <2021-08-24 17:42:46 (otoolej)>
%-------------------------------------------------------------------------------
function [g2,P,Ph_floor,Nfreq]=gen_lag_kern(win_params,N,Nfreq)
if(nargin<3 || isempty(Nfreq)), Nfreq=[]; end


DBplot=0;


%---------------------------------------------------------------------
% 1. make sure parameters are ok
%---------------------------------------------------------------------
P=win_params{1};
% window length can not be greater than N:
if(P>N)
    warning('length of g2 too long: chopping down length N');
    P=N;
end
% force window length to be odd-value:
if(P/2==fix(P/2))
    P=P-1; 
    warning(['Forcing P to be odd. P is now P=',num2str(P)]);
end
Ph_floor=floor(P/2);


l=length(win_params);
if( l<2 ) 
  error('Need at least two window parameters'); 
end

if(isempty(Nfreq)), Nfreq=P+1; end

if(Nfreq<(P+1))
    warning('Nfreq is too short: increasing length to (P+1)');
    Nfreq=P+1;
end

if(rem(Nfreq,2))
    Ntime=Ntime+1; 
    warning(['Forcing Ntime to be even. Ntime is now Ntime=',num2str(Ntime)]);
end


%---------------------------------------------------------------------
% 2. call function to generate window
%---------------------------------------------------------------------
win_type=win_params{2}; 
if( l>=3 ) win_extra_param=win_params{3}; else, win_extra_param=[]; end
if( l>3 )  win_dft_param=win_params{4}; else, win_dft_param=0; end


g2=get_window(P,win_type,win_extra_param,win_dft_param);
g2=g2(:);

g2_pad=padWin(g2,Nfreq);


%---------------------------------------------------------------------
% 3. [OPTIONAL] post-process window
%---------------------------------------------------------------------
% a. if normalizing window:
if(length(win_params)>4)
  if(strcmpi(win_params{5},'y')==1)
    g2=g2./sum(g2);
  end
end

% b. if (time) reversing window
if(length(win_params)>5 )
  g2=shiftWin(g2);
  g2_pad=shiftWin(g2_pad);
end


if(DBplot)
    figure(20); clf; hold all;
    subplot(211); plot(g2_pad);
    subplot(212); plot(g2);    
end


function w=shiftWin(w)
%---------------------------------------------------------------------
% shift window to centre
%---------------------------------------------------------------------
w=circshift(w(:),ceil(length(w)/2));





