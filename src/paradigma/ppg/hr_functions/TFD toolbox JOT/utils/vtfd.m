%-------------------------------------------------------------------------------
% Plot TFD and time/frequency plots
%
% USE:  vtfd(tfd,sig,Fs)
%
%  INPUT:
%        tfd    - the time-frequency distribution 
%        sig    - time-domain signal
%        Fs     - sampling frequency

%
%   Copyright (c) 2010, John M. O' Toole, The University of Queensland
%   All rights reserved.
%
%  Redistribution and use in source and binary forms, with or without
%  modification, are permitted provided that the following
%  conditions are met:
%      * Redistributions of source code must retain the above
%        copyright notice, this list of conditions and the following
%        disclaimer.
%      * Redistributions in binary form must reproduce the above
%        copyright notice, this list of conditions and the following
%        disclaimer in the documentation and/or other materials
%        provided with the distribution.
%      * Neither the name of the The University of Queensland nor the 
%        names of its contributors may be used to endorse or promote 
%        products derived from this software without specific prior 
%        written permission.
%  
%  THIS SOFTWARE IS PROVIDED BY JOHN M. O' TOOLE ''AS IS'' AND ANY
%  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
%  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
%  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL JOHN M. O' TOOLE BE
%  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
%  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
%  OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
%  BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
%  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
%  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
%  USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
%  DAMAGE.
%-------------------------------------------------------------------------------
function [h_image,h_time,h_freq]=vtfd(tfd,s1,FS,n,k)
if(nargin<3 || isempty(FS)) 
  FS=1; 
end
if(nargin<2 || isempty(s1)) 
  TFD_ONLY=1; 
  Ntime=size(tfd,1);  
else 
  TFD_ONLY=0; 
  Ntime=length(s1);
end
if(nargin<4 || isempty(n)), n=[]; end
if(nargin<5 || isempty(k)), k=[]; end


[N,M]=size(tfd);


% $$$ clf;


%---------------------------------------------------------------------
% Frame sizes:
%---------------------------------------------------------------------
X_AXIS_WIDTH=0.75;
Y_AXIS_HEIGHT=0.75;
X_AXIS_START=0.23;
Y_AXIS_START=0.23;
TIME_FREQ_PLOTS_WIDTH=0.15;
TIME_FREQ_PLOTS_GAP=0.07;


%---------------------------------------------------------------------
% X/Y tick labels
%---------------------------------------------------------------------
if(isempty(n))
    ntime=1:Ntime; ntime=ntime./FS;
    n=linspace(ntime(1),ntime(end),N);
else
    ntime=n;
end
Mh_time=ceil(Ntime/2);  Mh=ceil(M/2);
if(isempty(k))

    k=linspace(0,0.5,Mh);
    k_time=linspace(0,0.5,Mh_time);
    k=k.*FS;
else
    k_time=k;
end
% $$$ keyboard;

if(TFD_ONLY)
    imagesc(k,n,tfd); axis('xy');
    
    % seems to bug with Octave and gnuplot, so have to force square:
    if(is_octave), axis('square');  end
  
else
  %---------------------------------------------------------------------
  % 1. time plot
  %---------------------------------------------------------------------
  s1=real(s1); 
  h_time=subplot(2,2,1);
  set(h_time,'position',[(X_AXIS_START-TIME_FREQ_PLOTS_WIDTH-TIME_FREQ_PLOTS_GAP) ...
                      Y_AXIS_START TIME_FREQ_PLOTS_WIDTH ...
                      Y_AXIS_HEIGHT]);
  plot(s1,1:length(s1)); 
  axis('tight');
  grid('on');
  set(h_time,'xticklabel',[]); set(h_time,'yticklabel',[]);


  %---------------------------------------------------------------------
  % 2. freq plot
  %---------------------------------------------------------------------
  S1=fft(s1);
  h_freq=subplot(2,2,4);
  set(h_freq,'position',[X_AXIS_START (Y_AXIS_START- ...
                                       TIME_FREQ_PLOTS_WIDTH-TIME_FREQ_PLOTS_GAP) ...
                      X_AXIS_WIDTH TIME_FREQ_PLOTS_WIDTH]);
  plot(1:Mh_time,abs(S1(1:Mh_time)).^2);
  axis('tight');
  grid('on');
  set(h_freq,'xticklabel',[]); set(h_freq,'yticklabel',[]);


  %---------------------------------------------------------------------
  % 2. time-freq plot
  %---------------------------------------------------------------------
  h_image=subplot(2,2,2);
  set(h_image,'position',[X_AXIS_START Y_AXIS_START X_AXIS_WIDTH ...
                      Y_AXIS_HEIGHT]);

  imagesc(k,n,tfd); axis('xy');
  
  % seems to bug with Octave and gnuplot, so have to force square:
  if(is_octave), axis('square');  end
end


function r = is_octave ()
  persistent x;
  if (isempty (x))
    x = exist ('OCTAVE_VERSION', 'builtin');
  end
  r = x;
