%-------------------------------------------------------------------------------
% Generate a linear frequency modulated (LFM) signal
%
%  USE: sig=gen_LFM(N,fstart,fstop,phase)
%
%  INPUT:
%       N      = length of LFM signal
%       fstart = starting frequency
%       fend   = ending frequency
%       phase  = (optional) phase offset
%
%  OUTPUT:
%       sig = LFM signal of length-N
%
%  EXAMPLE:
%       N=256; fstart=0.1; fend=0.4;
%       x=gen_LFM(N,fstart,fend);
%       plot(x); 
%-------------------------------------------------------------------------------
function sig=gen_LFM(N,fstart,fstop,phase)
if(nargin<1 || isempty(N)) N=128; end
if(nargin<2 || isempty(fstart)) fstart=0.1; end
if(nargin<3 || isempty(fstop)) fstop=0.4; end
if(nargin<4 || isempty(phase)) phase=0; end



n=0:N-1;
sig=cos( 2*pi.*(fstart.*n + ((fstop-fstart)/(2*N)).*(n.^2)) + phase );
