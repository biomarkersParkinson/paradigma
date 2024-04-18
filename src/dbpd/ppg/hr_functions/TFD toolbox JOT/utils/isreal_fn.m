function flag = isreal_fn(fn,E)
% ISREAL_FN - is 'fn' real valued

if(nargin<2)
  E=sum(abs(fn(:)).^2) * 1e-12;
end
flag=1;

if( max( abs(imag(fn(:))) ) > E ) flag=0; end


