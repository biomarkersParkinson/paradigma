%-------------------------------------------------------------------------------
% gen_Doppler_lag_kern: Doppler--lag kernel (for GDTFD definition)
%
% Syntax: kern=gen_Doppler_lag_kern(kern_type,kern_params,N)
%
% Input: 
%   kern_type = { 'wvd' | 'swvd' | 'pwvd' | 'sep' | 'cw' | 'mb'}
%            wvd  - kernel for Wigner-Ville distribution
%            swvd - kernel for smoothed Wigner-Ville distribution
%                   (lag-independent kernel)
%            pwvd - kernel for pseudo Wigner-Ville distribution
%                   (Doppler-independent kernel)
%            sep  - kernel for separable kernel (combintation of SWVD and PWVD)
%            cw   - kernel for Choi-Williams distribution
%            mb   - kernel for Modified-B distribution
% 
%   kern_params = cell of kernel parameters:
%            wvd  - {}
%            swvd - {win_length,win_type,[win_param],[Doppler_or_time]}
%                   e.g. {11,'hamm',0,1}
%                   the parameter Doppler_or_time is either
%                       0 = to define window in the time-domain
%                       1 = to define window in the Doppler-domain
%            pwvd - {win_length,win_type,[win_param]}
%                   e.g. {200,'cosh',0.1}
%            sep  - { {win1_length,win1_type,[win1_param]}, 
%                    {win2_length,win2_type,[win2_param]}
%                   where win1 is the Doppler window and win2 is the lag window.
%                   e.g. { {11,'hamm'}, {200,'cosh',0.1} }
%            cw   - {sigma_parameter}
%            mb   - {beta_parameter} in the range 1<beta<0
%
%   N = signal length.
%
% Output:
%   g = Doppler--lag smoothing kernel
%
% See also: GEN_LAG_KERN, GEN_DOPPLER_KERN, GET_WINDOW, PADWIN
%
% Example:
%      N=128; 
%      g=gen_Doppler_lag_kern( 'sep',{{N-1,'cosh',0.1,1},{51,'hamm'}},N); 
%      clf; mesh( fftshift(g) );
%      xlabel('lag'); ylabel('Doppler');
%      title('Separable kernel');
%

% John M. O' Toole, University College Cork
% Started: 14-04-2014
%
% last update: Time-stamp: <2018-01-21 04:51:27 (otoolej)>
%-------------------------------------------------------------------------------
function g=gen_Doppler_lag_kern(kern_type,kern_params,N,lag_index,G1)
if(nargin<3), error('need 3 input arguments'); end
if(nargin<4 || isempty(lag_index)), lag_index=[]; end
if(nargin<5 || isempty(G1)), G1=[]; end



DBplot=0;

lag_sample=2;
doppler_sample=1/N;

if(isempty(lag_index))
    lag_index=1:N;
    g=zeros(N);
else
    g=zeros(N,1:length(lag_index));    
end



if(~iscell(kern_params))
    tmp_params{1}=kern_params; 
    kern_params=tmp_params;
end

g=get_kern(g,lag_index,kern_type,kern_params,doppler_sample,lag_sample,N,G1);


% All kernels are real valued. 
g=real(g);

if(DBplot)
    figure(1); clf; 
    if(length(lag_index)==1)
        plot( g );
    else
        mesh( fftshift(g) );
    end
    xlabel('lag'); ylabel('Doppler');
    title('Separable kernel');
end




function g=get_kern(g,lag_index,kernel_type,kernel_params,doppler_sample_rate, ...
                      lag_sample_rate,N,G1)
%---------------------------------------------------------------------
% generate the kernels for a given sample rate
%---------------------------------------------------------------------
if(nargin<8 || isempty(G1)), G1=[]; end

[Nd,Nl]=size(g);
l=length(kernel_params);


switch kernel_type
  
 case {'wvd'}
  g(:,:)=1;

  
  %---------------------------------------------------------------------
  % Smoothed Wigner-Ville (Lag Independent (LI) kernel)
  % g(l/NT,mT) = W(l/NT)
  %---------------------------------------------------------------------
 case 'swvd'
  
  if( l<2 ) 
    error('Need at least two window parameters for SWVD'); 
  end
  win_length=kernel_params{1};
  win_type=kernel_params{2}; 
  win_param=0; win_param2=1;
  if( l>=3 ) win_param=kernel_params{3}; end
  if( l>=4 ) win_param2=kernel_params{4}; end
  

  G1=get_window(win_length,win_type,win_param);
  G1=padWin(G1,Nd);
  
  
  % Define window in the time-domain (this is the usual practice):
  % But could also define in the doppler-domain.
  if( win_param2==0 )
    G1=fft(G1);
    G1=G1./G1(1);
  end
  
  if( isreal_fn(ifft(G1))==0 )
    warning('Window function g1(t) is NOT real valued.');
  end
  
  for m=1:Nl
      g(:,m) = G1;
  end
  
  
  %---------------------------------------------------------------------
  % Pseudo-Wigner-Ville (Doppler Independent (DI) kernel)
  % g(l/NT,mT) = g2(mT)
  %---------------------------------------------------------------------
 case 'pwvd'
  
  if( l<2 ) 
    error('Need at least two window parameters for PWVD'); 
  end
  P=kernel_params{1};
  win_type=kernel_params{2}; 
  win_param=0;   win_param2=0;
  if( l>2 ) win_param=kernel_params{3}; end
  if( l>3 ) win_param2=kernel_params{4}; end  

  g2=get_window(P,win_type,win_param,win_param2);

  g2=padWin(g2(:),N);
  g2=g2(lag_index);
    
  if( Nd==Nl && isreal_fn(fft(g2))==0 )
    warning('Window function G2(f) is NOT real valued.');
  end

  for l=0:Nd-1
      g(l+1,:)=g2;
  end

  %---------------------------------------------------------------------
  % Seperable Kernel
  %
  % g(l/NT,mT) = G1(l/NT)g2(mT) 
  %  
  %---------------------------------------------------------------------
 case { 'sep', 'sep-full' }

  if(l<2)
    error('Need at least two windows parameters.'); 
  end
    
  g1=get_kern(g,lag_index,'swvd',kernel_params{1},doppler_sample_rate,lag_sample_rate,N);
  g2=get_kern(g,lag_index,'pwvd',kernel_params{2},doppler_sample_rate,lag_sample_rate,N);
  g=g1.*g2;
  
  
  %---------------------------------------------------------------------
  % Choi-Williams
  %
  % g(l/NT,mT) = exp( -(2 pi m l/N)^2/ sigma ) 
  %
  %---------------------------------------------------------------------
 case {'cw', 'choi-williams', 'choi-will'}
  
  if(l>=1) 
    sigma=kernel_params{1};
  else
    sigma=11;
  end
  Ndh=ceil(Nd/2);
  Nlh=ceil(Nl/2);
  

  % if just at one-lag index:
  if(length(lag_index)==1)
    g(1)=1;
    m=lag_index-1;
    if(m==(N-1))
        g(:)=0;
    elseif(m==0)
        g(:)=1;
    else
        if(m>ceil(N/2)) m=N-m; end

        const = ((2*pi)^2)/sigma;
        u=1:Ndh-1;
        u1=u.*doppler_sample_rate; m1=m*lag_sample_rate;
   
        g(u+1) = exp( -const.*(u1.*m1).^2 ).';
        g(Nd-u+1)=g(u+1); 
    end

    % or for the whole thing: 
  else
      g(1,1:Nl)=1; g(1:Nd,1)=1;  
      
      const = ((2*pi)^2)/sigma;
      u=1:Ndh-1;
      for m=1:Nlh-1
          
          u1=u.*doppler_sample_rate; m1=m*lag_sample_rate;
          g(u+1,m+1)=exp( -const.*(u1.*m1).^2 ).';
          
          g(Nd-u+1,m+1)=g(u+1,m+1); 
          g(u+1,Nl-m+1)=g(u+1,m+1); 
          g(Nd-u+1,Nl-m+1)=g(u+1,m+1); 
      end
  end
  
  %---------------------------------------------------------------------
  % Product kernel (sometimes called a RID kernel)
  %
  % g(l/NT,2mT)=H(l2m/N)
  %---------------------------------------------------------------------
  case { 'prod', 'RID', 'product' }

    % NOT WORKING:
% $$$     if(length(lag_index)==1)
% $$$         m=lag_index;
% $$$         
% $$$         g(1)=1;
% $$$         if(m==N)
% $$$             g(:)=0;
% $$$         else
% $$$             if(m>ceil(N/2)) m=N-m; end
% $$$             
% $$$             u=1:floor(Nd/2);
% $$$             im=mod(u*m,length(G1));
% $$$             
% $$$             g(u+1)=G1(im+1);
% $$$             g(Nd-u+1)=g(u+1); 
% $$$         end
% $$$     else
% $$$         keyboard;
% $$$         if(isempty(G1))
% $$$             [d,d2,d3,G1]=gen_Doppler_kern(kernel_params,N,N);
% $$$         end
% $$$         Ndh=ceil(Nd/2);  Nlh=ceil(Nl/2);
% $$$         g=zeros(Nd,Nl);
% $$$ 
% $$$         g(1,1:Nl)=G1(1); g(1:Nd,1)=G1(1);  
% $$$ 
% $$$         u=1:Ndh-1;
% $$$         for m=1:Nlh-1
% $$$             g(u+1,m+1)=G1( mod(u.*m,N)+1 );
% $$$             
% $$$ % $$$             g(Nd-u+1,m+1)=g(u+1,m+1); 
% $$$ % $$$             g(u+1,Nl-m+1)=g(u+1,m+1); 
% $$$ % $$$             g(Nd-u+1,Nl-m+1)=g(u+1,m+1); 
% $$$         end
% $$$         
% $$$     end
    
   
  %---------------------------------------------------------------------
  % Modified-B (a specific SWVD or LAG INDEPENDENT kernel)
  %
  % G(nT,mT) = cosh^{-2*beta}(n)   (defined in time--lag domain)
  %---------------------------------------------------------------------
 case { 'mb', 'modified-b' }

  G1=get_kern(g,lag_index,'swvd',{N-1,'cosh',kernel_params{1}}, ...
                doppler_sample_rate,lag_sample_rate,N);   

  for m=1:Nl
      g(:,m) = G1;
  end

  
 otherwise
  error(['Unknown kernel type: ' kernel_type]);
  
end

