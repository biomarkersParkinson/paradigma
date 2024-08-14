%-------------------------------------------------------------------------------
% fold_vector_half: Fold vector
%
%    y[n] = y[n] + sum_{p=0}^{a-l} x[pJ+n]
%
%    for n=0,1,...,Jh
%
% Syntax: y=fold_vector_half(x,J,Jh,a)
%
% Inputs: 
%     x,J,Jh,a - 
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
% last update: Time-stamp: <2014-04-23 14:05:04 (otoolej)>
%-------------------------------------------------------------------------------
function y=fold_vector_half(x,J,Jh,a)
% Assume that all parameters (J,Jh,a) are sane.

y=zeros(Jh+1,1);
x=x(:);

n=0:(Jh);
for p=0:a-1
  y(n+1)=y(n+1)+x(p*J+n+1);    
end
