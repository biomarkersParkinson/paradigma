%-------------------------------------------------------------------------------
% sep_gdtfd: Time--frequency distribution (quadratic class) with separable kernel of
% the form g[l,m]=G₁[l]g₂[m]
%
% Syntax: tfd=sep_gdtfd(x,dopp_win_params,lag_win_params,Ntime,Nfreq)
%
% Inputs: 
%      x = input signal (either real-valued signal of length-N or
%          complex-valued analytic signal of length-2N)
%
%      dopp_win_params = Doppler window parameters in cell form:
%                 {win_length,win_type,win_param,Doppler_or_not} where
%                     - win_length is the sample length of the window
%                     - win_type is the type of window 
%                     - [optional] win_param is the parameter of the window 
%                     - [optional] Doppler_or_not is either 1 (define window in the Doppler
%                     domain, which is the default) or 0 (define window the time domain)
%                 e.g. {121, 'hamm'}; {121, 'tukey', 0.2}; {127,'cosh',0.01,0}
%
%      lag_win_params = lag window parameters in cell form:
%                 {win_length,win_type,win_param,lag_or_not} where
%                     - win_length is the sample length of the window
%                     - win_type is the type of window 
%                     - [optional] win_param is the parameter of the window 
%                     - [optional] lag_or_not is either 0 (define window in the lag
%                     domain, which is the default) or 0 (define window the frequency domain)
%                 e.g. {121, 'hamm'}; {121, 'tukey', 0.2}; {127,'cosh',0.01,0}
%
%      Nfreq = frequency oversampling value; must be greater than length of lag window
%      Ntime = time oversampling value; must be greater than length of Doppler window
%
% Outputs: 
%      tfd = Ntime x Nfreq time-frequency distribution
%
% See also: DEC_SEP_GDTFD, GET_ANALYTIC_SIGNAL, GEN_LAG_KERN, GEN_DOPPLER_KERN, FFT
%
% Example:
%      N=10000; Ntime=256; Nfreq=256;
%      x=gen_LFM(N,0.1,0.3)+gen_LFM(N,0.4,0.1);
%
%      c=sep_gdtfd(x,{51,'hann'},{171,'hann'},Ntime,Nfreq); 
%      vtfd(c,x);


% John M. O' Toole, University College Cork
% Started: 16-04-2014
%
% last update: Time-stamp: <2019-02-19 15:12:35 (otoolej)>
%-------------------------------------------------------------------------------
function tfd=sep_gdtfd(x,dopp_win_params,lag_win_params,Ntime,Nfreq)
if(nargin<4 || isempty(Ntime)), Ntime=[]; end
if(nargin<5 || isempty(Nfreq)), Nfreq=[]; end

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
% 2. compute lag and Doppler windows
%---------------------------------------------------------------------
[g2,P,Ph_floor,Nfreq]=gen_lag_kern(lag_win_params,N,Nfreq);
Nh_freq=ceil(Nfreq/2);
Ph=ceil(P/2);
[G1,Q,Ntime,G1_pad]=gen_Doppler_kern(dopp_win_params,N,Ntime);
Nh_time=ceil(Ntime/2);
Qh=ceil(Q/2);


%---------------------------------------------------------------------
% 3. convolve signal function with kernel (multiplication in the 
%    Doppler--lag domain). Do this in stages to minimise memory
%---------------------------------------------------------------------
if(DBmem), s=whos; fprintf('start: mem=%s\n',disp_bytes(sum([s.bytes]))); end
tfd=zeros(Ntime,Nfreq); 

if(DBmem), s=whos; fprintf('tfd declare: mem=%s\n',disp_bytes(sum([s.bytes]))); end

n=0:N-1; l=0:(Qh-1); lb=1:(Qh-1);
for m=0:(Ph-1);
    inp=mod(n+m,N2);  inn=mod(n-m,N2); 
    K_lag_slice=g2(m+1).*z(inp+1).*conj( z(inn+1) );
    
    K_lag_slice=fft(K_lag_slice);
    
    R_lag_slice=zeros(Ntime,1);
    R_lag_slice(l+1)=K_lag_slice(l+1).*G1(l+1);
    R_lag_slice(Ntime-lb+1)=K_lag_slice(N-lb+1).*G1(Q-lb+1);
    
    R_lag_slice=ifft(R_lag_slice);
    
    tfd(:,m+1)=real( R_lag_slice );
    tfd(:,m+Nh_freq+1)=imag( R_lag_slice );    
end

if(DBmem), s=whos; fprintf('R: mem=%s\n',disp_bytes(sum([s.bytes]))); end



if(DBcompare)
    K=zeros(N,Nh_freq);
    m=0:(Ph-1);
    for n=0:N-1
        inp=mod(n+m,N2);  inn=mod(n-m,N2); 
        K(n+1,m+1)=g2(m+1).*z(inp+1).*conj( z(inn+1) );
    end
    af=fft(K);

    R=zeros(Ntime,Nh_freq);
    for m=0:Ph-1
        R(l+1,m+1)=af(l+1,m+1).*G1(l+1);
        R(Ntime-lb+1,m+1)=af(N-lb+1,m+1).*G1(Q-lb+1);
        
        R(:,m+1)=ifft(R(:,m+1));
    end
    Rtest=complex(tfd(:,1:Nh_freq),tfd(:,(Nh_freq+1):end));
    dispEE(R,Rtest);
    clear Rtest K af;
end


%-------------------------------------------------------------------------
% 4.  Expand R for positive and negative lag values and DFT back to 
%     time--frequency domain
%-------------------------------------------------------------------------
m=0:(Ph-1); mb=1:(Ph-1);
k_real=1:Nh_freq; k_imag=(Nh_freq+1):Nfreq;

for n=0:2:Ntime-2
    R_even_half=complex(tfd(n+1,k_real),tfd(n+1,k_imag));
    R_odd_half =complex(tfd(n+2,k_real),tfd(n+2,k_imag));    
    
    R_tslice_even=zeros(1,Nfreq);  R_tslice_odd=zeros(1,Nfreq);
    R_tslice_even(m+1)=R_even_half(m+1);
    R_tslice_odd(m+1) =R_odd_half(m+1);
    R_tslice_even(Nfreq-mb+1)=conj( R_even_half(mb+1) );
    R_tslice_odd(Nfreq-mb+1) =conj( R_odd_half(mb+1) );
    
    tfd_time_slice=fft( R_tslice_even+j.*R_tslice_odd );

    tfd(n+1,:)=real(tfd_time_slice);
    tfd(n+2,:)=imag(tfd_time_slice);
end

if(DBmem), s=whos; fprintf('tfd (end): mem=%s\n',disp_bytes(sum([s.bytes]))); end


if(DBcompare)
    Rfull=zeros(Ntime,Nfreq);
    Rfull(:,m+1)=R(:,m+1); 
    Rfull(:,Nfreq-mb+1)=conj( Rfull(:,mb+1) );

    tfd_test=fft( Rfull.' ).';
    dispEE(tfd_test,tfd);
    clear R test_test;
end

scale_factor=1/Nfreq;
tfd=tfd.*scale_factor;


%---------------------------------------------------------------------
% END; testing and plotting
%---------------------------------------------------------------------
if(DBtime),  dispVars( toc(time_start) ); end

if(DBtest) % && Ntime==N && Nfreq==N)
    if(DBtime)
        dispVars( toc(time_start) );
        time_start=tic;
    end
    tfd_test=full_gdtfd_testing_version(x,'sep',{dopp_win_params,lag_win_params});
    
    if(DBtime), dispVars( toc(time_start) ); end
    
% $$$     dispVars( sum(tfd_test(:)), sum(tfd(:)) );
    if(Ntime==N && Nfreq==N)
        dispEE(tfd_test,tfd);
    else
        a=N/Ntime; b=N/Nfreq; 
        if( a==floor(a) && b==floor(b) )
            dispEE(tfd_test(1:a:end,1:b:end),tfd./(a*b));
        end
    end
end
if(DBplot)
    figure(1); clf; 
    vtfd(tfd,real(x(1:N)));
end


