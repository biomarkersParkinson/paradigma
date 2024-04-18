%-------------------------------------------------------------------------------
% fold_vector_full: fold the vector for frequency decimation:
%
%    y[n] = y[n] + sum_{p=0}^{a-l} x[pJ+n]
%
%    for n=0,1,...,J-1
%
%    which returns DFT{y[n]} = Y[ak]
%
% Syntax: y=fold_vector_full(x,J,a)
%
% Inputs: 
%     x,J,a - 
%
% Outputs: 
%     y - 
%
% Example:
%     
%

% John M. O' Toole, University College Cork
% Started: 23-04-2014
%
% last update: Time-stamp: <2014-04-23 17:11:54 (otoolej)>
%-------------------------------------------------------------------------------
function y=fold_vector_full(x,J,a)
% Assume that all parameters (J,a) are sane.

y=zeros(J,1);
x=x(:);

n=0:J-1;
y(n+1)=x(n+1);    
for p=1:a-1
  y(n+1)=y(n+1)+x(p*J+n+1);    
end

% scale by decimation factor:
y=y./a;
