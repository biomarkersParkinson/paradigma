%--------------------------------------------------------------------------------
%
% Pad window to Npad. 
%
% Presume that positive window indices are first. 
%
%  USE: w_pad=padWin(w,Npad)
%
%  INPUT: 
%        w    = window (vector) of length N
%        Npad = pad window to length N (Npad>N)
%
%  OUTPUT: 
%        w_pad = window of length N zeropadded to length Npad
%
%
% When N is even use method described in [1]
%
%   References:
%     [1] S. Lawrence Marple, Jr., Computing the discrete-time analytic
%     signal via FFT, IEEE Transactions on Signal Processing, Vol. 47,
%     No. 9, September 1999, pp.2600--2603.
%
%
% STARTED: 04-01-2008

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
%--------------------------------------------------------------------------------
function w_pad=padWin(w,Npad)
DB=0;
w=w(:);
w_pad=zeros(Npad,1);
N=length(w);
Nh=floor(N/2);
if(DB) dispVars(N,Npad); end
if(Npad<N) error('Npad is less than N'); end


% Trival case:
if(N==Npad) 
  w_pad=w; 
  return; 
end


 % For N odd:
if( rem(N,2)==1 )
  if(DB) dispDB('odd'); end
 n=0:Nh;
 w_pad(n+1)=w(n+1); 
 n=1:Nh;
 w_pad(Npad-n+1)=w(N-n+1);
 
 % For N even:
 % split the Nyquist frequency in two and distribute over positive
 % and negative indices.
else
  if(DB) dispDB('even'); end
 n=0:(Nh-1);
 w_pad(n+1)=w(n+1); 
 w_pad(Nh+1)=w(Nh+1)/2; 

 n=1:Nh-1;
 w_pad(Npad-n+1)=w(N-n+1);
 w_pad(Npad-Nh+1)=w(Nh+1)/2; 
end


