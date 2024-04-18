%-------------------------------------------------------------------------------
% gen_Doppler_kern: smoothing window for Doppler-kernel G1[l]; checks paramaters
%
% Syntax: [G1,Q,Qh_floor,Ntime]=gen_Doppler_kern(win_params,N,Ntime)
%
%  Input:
%       win_params - cell of {win_length,win_type,[win_param],[Doppler_or_time]}
%                     e.g. {11,'hamm',0,1}
%                     the parameter 'Doppler_or_time' is either:
%                       0 = to define window in the time-domain
%                       1 = to define window in the Doppler-domain (default)
%       N          - length of signal
%       Ntime      - number of sample points in the time direction
%
%  Ouput:
%       G1           - Doppler function G_1(1/(QT))
%       Q            - length of G1
%       Ntime        - returns Ntime as it may be adjusted from input
%
%  See also: GEN_LAG_KERN, GEN_DOPPLER_LAG_KERN, GET_WINDOW, PADWIN
%
%  Example:
%       G1=gen_Doppler_kern({111,'cosh',0.1,1},128,512);
%       plot(real(G1));
%
% 
% IF window is defined in doppler domain:
%     window function G_1(l/NT) is length Q
% OR if window is defined in time domain:
%     window function g_1(nT) is length-Q, and then zero-padded 
%     to the doppler domain where G_1(l/NT) is length N
% ENDIF 

% John M. O' Toole, University College Cork
% Started: 16-04-2014
%
% last update: Time-stamp: <2021-08-24 17:42:57 (otoolej)>
%-------------------------------------------------------------------------------
function [G1,Q,Ntime,G1_pad]=gen_Doppler_kern(win_params,N,Ntime)
if(nargin<3 || isempty(Ntime)), Ntime=[]; end


DBplot=0;


%---------------------------------------------------------------------
% 1. make sure parameters are ok
%---------------------------------------------------------------------
Q=win_params{1};
% window length can not be greater than N:
if(Q>N)
    warning('length of g2 too long: chopping down length N');
    Q=N;
end
% force window length to be odd-value:
if(Q/2==fix(Q/2))
    Q=Q-1; 
    warning(['Forcing Q to be odd. Q is now Q=',num2str(Q)]);
end
Qh_floor=floor(Q/2);


l=length(win_params);
if( l<2 ) 
  error('Need at least two window parameters'); 
end
if(isempty(Ntime)), Ntime=Q+1; end

if(Ntime<(Q+1))
    warning('Ntime is too short: increasing length to (Q+1)');
    Ntime=Q+1;
end

% force Ntime to be even
if(rem(Ntime,2)), 
    Ntime=Ntime+1; 
    warning(['Forcing Ntime to be even. Ntime is now Ntime=',num2str(Ntime)]);
end



%---------------------------------------------------------------------
% 2. call function to generate window
%---------------------------------------------------------------------
win_type=win_params{2}; 
if( l>=3 ) win_extra_param=win_params{3}; else, win_extra_param=[]; end
if( l>3 )  win_dft_param=win_params{4}; else, win_dft_param=1; end


% if define in Doppler domain:
if(win_dft_param==1)
    G1=get_window(Q,win_type,win_extra_param);
    G1=G1(:);

    G1_pad=padWin(G1,Ntime);
    
    
else
    % or defining window in the time-domain:
    g1=get_window(Q,win_type,win_extra_param,0);
    g1=padWin(g1,N);
    G1_length=N;

    G1=real( fft(g1) ); 
    G1=G1./G1(1);
    Ntime=N; Q=N;
end


%---------------------------------------------------------------------
% 3. [OPTIONAL] post-process window
%---------------------------------------------------------------------
% a. if normalizing window:
if(length(win_params)>4)
  if(strcmpi(win_params{5},'y')==1)
    G1=G1./sum(G1);
  end
end

% b. if (time) reversing window
if(length(win_params)>5 )
  G1=shiftWin(G1);
  G1_pad=shiftWin(G1_pad);
end


if(DBplot)
    figure(20); clf; hold all;
    if(exist('G1_pad','var'))
        subplot(211); plot(G1_pad);
    end
    subplot(212); plot(real(G1));    
end




function w=shiftWin(w)
%---------------------------------------------------------------------
% shift window to centre
%---------------------------------------------------------------------
w=circshift(w(:),ceil(length(w)/2));
