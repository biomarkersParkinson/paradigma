%-------------------------------------------------------------------------------
% nonsep_gdtfd: Time-frequency distribution (quadratic class) with non-separable kernel
%
% Syntax: [tfd,g]=nonsep_gdtfd(x,kern_type,kern_params)
%
% Inputs: 
%      x = input signal (either real-valued signal of length-N or
%          complex-valued analytic signal of length-2N)
%
%      kern_type = { 'wvd' | 'swvd' | 'pwvd' | 'sep' | 'cw' | 'mb' }
%            wvd  - Wigner-Ville distribution
%            swvd - Smoothed Wigner-Ville distribution
%                   (lag-independent kernel)
%            pwvd - Pseudo Wigner-Ville distribution
%                   (Doppler-independent kernel)
%            sep  - Separable-kernel distribution 
%                   (combintation of SWVD and PWVD)
%            mb   - Modified-B distribution
%            cw   - Choi-Williams distribution
% 
%      kern_params = cell of kernel parameters:
%            wvd  - {}
%            swvd - {win_length,win_type,[win_param]}
%                   e.g. {11,'hamm'}
%            pwvd - {win_length,win_type,[win_param]}
%                   e.g. {200,'cosh',0.1
%            sep  - { {win1_length,win1_type,[win1_param]}, 
%                    {win2_length,win2_type,[win2_param]}
%                   where win1 is the doppler window and win2 is the 
%                   lag window, e.g. { {11,'hamm'}, {200,'cosh',0.1} }
%            mb   - {beta_parameter} in the range 1<beta<0
%            cw   - {sigma_parameter}
%
% Outputs: 
%     tfd = N x N time-frequency distribution
%
% See also: DEC_NONSEP_GDTFD, GET_ANALYTIC_SIGNAL, GEN_DOPPLER_LAG_KERN, FFT
%
% Example:
%      N=128; 
%      x=gen_LFM(N,0.1,0.3)+gen_LFM(N,0.4,0.1);
%
%      c=nonsep_gdtfd(x,'cw',{100}); 
%      vtfd(c,x);


% John M. O' Toole, University College Cork
% Started: 14-04-2014
%
% last update: Time-stamp: <2016-08-25 17:39:48 (otoolej)>
%-------------------------------------------------------------------------------
function tfd=nonsep_gdtfd(x,kern_type,kern_params)
if(nargin<2 || isempty(kern_type)), kern_type='cw'; end
if(nargin<3 || isempty(kern_params)), kern_params=10; end


DBplot=0;
DBmem=0;
DBcompare=0;
DBtest=0;
DBtime=0;


if(DBtime), time_start=tic; end
%---------------------------------------------------------------------
% 1. convert real-valued signal to analytic signal of length 2N
%---------------------------------------------------------------------
[z,N2,N,Nh]=get_analytic_signal(x);


%---------------------------------------------------------------------
% 2. generate time--lag signal function (for positive-lag values only)
%---------------------------------------------------------------------
if(DBmem), s=whos; fprintf('start: mem=%s\n',disp_bytes(sum([s.bytes]))); end
tfd=zeros(N,N); 
m_real=1:Nh; m_imag=(Nh+1):N;

m=0:(Nh-1);
for n=0:N-1
    inp=mod(n+m,N2);  inn=mod(n-m,N2); 
    K_time_slice=z(inp+1).*conj( z(inn+1) );
    
    tfd(n+1,m_real)=real( K_time_slice );
    tfd(n+1,m_imag)=imag( K_time_slice );    
end

if(DBcompare)
    for n=0:N-1
        inp=mod(n+m,N2);  inn=mod(n-m,N2); 
        K(n+1,m+1)=z(inp+1).*conj( z(inn+1) );
    end
    Ktest=complex(tfd(:,m_real),tfd(:,m_imag));
    dispEE(Ktest,K);
end


if(DBmem), s=whos; fprintf('K: mem=%s\n',disp_bytes(sum([s.bytes]))); end

% $$$ if(strcmp(kern_type,'wvd'))
% $$$     for n=0:2:N-2
% $$$     
% $$$         tfd_time_slice=fft( R_tslice_even+j.*R_tslice_odd );
% $$$ 
% $$$         tfd(n+1,:)=real(tfd_time_slice);
% $$$         tfd(n+2,:)=imag(tfd_time_slice);
% $$$     end
% $$$     
% $$$     
% $$$     dispVars('here');
% $$$     return;
% $$$ end


%-------------------------------------------------------------------------
% 3. multiply kernel and signal function in the Doppler-lag domain
%-------------------------------------------------------------------------
for m=0:Nh-1
    g_lag_slice=gen_Doppler_lag_kern(kern_type,kern_params,N,m+1);
    
    R_lag_slice=ifft( fft( complex(tfd(:,m_real(m+1)),tfd(:,m_imag(m+1))) ).*g_lag_slice );

    
    tfd(:,m_real(m+1))=real( R_lag_slice );
    tfd(:,m_imag(m+1))=imag( R_lag_slice );    
end

if(DBcompare)
    g=gen_Doppler_lag_kern(kern_type,kern_params,N);
    R=ifft( fft(K).*g(:,1:Nh) );
    Rtest=complex(tfd(:,m_real),tfd(:,m_imag));
    dispEE(Rtest,R);
    clear g;
end


if(DBmem), s=whos; fprintf('R: mem=%s\n',disp_bytes(sum([s.bytes]))); end




%-------------------------------------------------------------------------
% 4.  Expand R for positive and negative lag values and DFT back to 
%     time--frequency domain
%-------------------------------------------------------------------------
m=0:(Nh-1); mb=1:(Nh-1);
for n=0:2:N-2
    R_even_half=complex(tfd(n+1,m_real),tfd(n+1,m_imag));
    R_odd_half =complex(tfd(n+2,m_real),tfd(n+2,m_imag));    
    
    R_tslice_even=zeros(1,N);  R_tslice_odd=zeros(1,N);
    R_tslice_even(m+1)=R_even_half(m+1);
    R_tslice_odd(m+1) =R_odd_half(m+1);
    R_tslice_even(N-mb+1)=conj( R_even_half(mb+1) );
    R_tslice_odd(N-mb+1) =conj( R_odd_half(mb+1) );
    
    tfd_time_slice=fft( R_tslice_even+j.*R_tslice_odd );

    tfd(n+1,:)=real(tfd_time_slice);
    tfd(n+2,:)=imag(tfd_time_slice);
end

if(DBmem), s=whos; fprintf('tfd: mem=%s\n',disp_bytes(sum([s.bytes]))); end

    
if(DBcompare)
    Rfull=zeros(N);
    Rfull(:,m+1)=R; 
    Rfull(:,N-mb+1)=conj( Rfull(:,mb+1) );

    tfd_test=fft( Rfull.' ).';
    dispEE(tfd_test,tfd);
end

if(DBmem), s=whos; fprintf('end: mem=%s\n',disp_bytes(sum([s.bytes]))); end
tfd=tfd./N;


if(DBtime), dispVars( toc(time_start) ); end

if(DBtest)
    if(DBtime), time_start=tic; end
    tfd_test=full_gdtfd_testing_version(x,kern_type,kern_params);
    if(DBtime), dispVars( toc(time_start) ); end
    dispEE(tfd_test,tfd);
end
if(DBplot)
    figure(1); clf; 
    vtfd(tfd,real(x(1:N)));
end

tfd = tfd';





