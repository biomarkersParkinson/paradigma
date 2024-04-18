%-------------------------------------------------------------------------------
% li_gdtfd: TFD with lag-independent (LI) kernel g[l,m]=G₁[l] 
%
% Syntax: tfd=li_gdtfd(x,dopp_win_params,Ntime)
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
%      Ntime = time oversampling value; must be greater than length of Doppler window
%
% Outputs: 
%      tfd = Ntime x N time-frequency distribution
%
% See also: DEC_LI_GDTFD, GET_ANALYTIC_SIGNAL, GEN_DOPPLER_KERN, FFT
%
% Example:
%      N=10000; Ntime=256; 
%      x=gen_LFM(N,0.1,0.3)+gen_LFM(N,0.4,0.1);
%
%      c=li_gdtfd(x,{51,'hamm'},Ntime); 
%      vtfd(c,x);
%
% Example:
%     
%

% John M. O' Toole, University College Cork
% Started: 16-04-2014
%
% last update: Time-stamp: <2014-07-23 15:28:16 (otoolej)>
%-------------------------------------------------------------------------------
function [tfd,G1]=li_gdtfd(x,dopp_win_params,Ntime)
if(nargin<2 || isempty(dopp_win_params)), dopp_win_params={11,'hamm',0,1}; end
if(nargin<3 || isempty(Ntime)), Ntime=[]; end


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
% 2. generate Doppler--frequency signal function 
%    (for positive-Doppler values only)
%---------------------------------------------------------------------
[G1,Q,Ntime]=gen_Doppler_kern(dopp_win_params,N,Ntime);
Nh=ceil(Ntime/2);
Qh=ceil(Q/2);

Z=fft(z);
if(DBmem), s=whos; fprintf('start: mem=%s\n',disp_bytes(sum([s.bytes]))); end
tfd=zeros(Ntime,N); 
l_real=1:Qh; l_imag=(Ntime-Qh+1):Ntime;


l=0:(Qh-1);
for k=0:N-1
    inp=mod(k+l,N2);  inn=mod(k-l,N2); 
    inpN=mod(k+l+N,N2);  innN=mod(k-l-N,N2);     
    K_doppler_slice=G1(l+1).*( Z(inp+1).*conj( Z(inn+1) )  + ...
                               Z(inpN+1).*conj( Z(innN+1) ) );
    
    tfd(l_real,k+1)=real( K_doppler_slice )./2;
    tfd(l_imag,k+1)=imag( K_doppler_slice )./2;    
end


% $$$ Ktesting=zeros(N,ceil(N/2)+1); m=0:(ceil(N/2));
% $$$ for n=0:N-1
% $$$     inp=mod(n+m,N2);  inn=mod(n-m,N2); 
% $$$     Ktesting(n+1,m+1)=z(inp+1).*conj( z(inn+1) );
% $$$ end
% $$$ Kfull=zeros(N); mb=1:(ceil(N/2)-1);
% $$$ Kfull(:,m+1)=Ktesting(:,m+1); 
% $$$ Kfull(:,N-mb+1)=conj( Ktesting(:,mb+1) );
% $$$ 
% $$$ Ktest=complex(tfd(l_real,:),tfd(l_imag,:));
% $$$ kk=fft(fft(Kfull.').');
% $$$ kk=kk(l+1,:).*2;
% $$$ dispVars(size(kk),size(Ktest));
% $$$ dispEE(Ktest,kk);
% $$$ figure(10); clf; vtfd(abs(Ktest));
% $$$ figure(11); clf; vtfd(abs(Ktest-kk));

if(DBcompare)
    K=zeros(Nh,N);
    for k=0:N-1
        inp=mod(k+l,N2);  inn=mod(k-l,N2); 
        inpN=mod(k+l+N,N2);  innN=mod(k-l-N,N2);     
        K(l+1,k+1)=G1(l+1).*( Z(inp+1).*conj( Z(inn+1) ) + ...
                              Z(inpN+1).*conj( Z(innN+1) ) );
    end
    Ktest=complex(tfd(l_real,:),tfd(l_imag,:));
    dispEE(Ktest,K(l+1,:));
end
if(DBmem), s=whos; fprintf('K: mem=%s\n',disp_bytes(sum([s.bytes]))); end




%-------------------------------------------------------------------------
% 3.  Expand R for positive and negative lag values and DFT back to 
%     time--frequency domain
%-------------------------------------------------------------------------
l=0:(Qh-1); lb=1:(Qh-1);
for k=0:2:N-2
    R_even_half=complex(tfd(l_real,k+1),tfd(l_imag,k+1));
    R_odd_half =complex(tfd(l_real,k+2),tfd(l_imag,k+2));    
    
    R_tslice_even=zeros(Ntime,1);  R_tslice_odd=zeros(Ntime,1);
    R_tslice_even(l+1)=R_even_half(l+1);
    R_tslice_odd(l+1) =R_odd_half(l+1);
    R_tslice_even(Ntime-lb+1)=conj( R_even_half(lb+1) );
    R_tslice_odd(Ntime-lb+1) =conj( R_odd_half(lb+1) );
    
    tfd_time_slice=ifft( R_tslice_even+j.*R_tslice_odd );

    tfd(:,k+1)=real(tfd_time_slice);
    tfd(:,k+2)=imag(tfd_time_slice);
end

if(DBmem), s=whos; fprintf('tfd: mem=%s\n',disp_bytes(sum([s.bytes]))); end



if(DBcompare)
    Rfull=zeros(Ntime,N);
    Rfull(l+1,:)=K(l+1,:); 
    Rfull(Ntime-lb+1,:)=conj( Rfull(lb+1,:) );

    tfd_test=ifft( Rfull );
    dispEE(tfd_test,tfd);
end


if(DBmem), s=whos; fprintf('end: mem=%s\n',disp_bytes(sum([s.bytes]))); end

scale_factor=1/N;
tfd=tfd.*scale_factor;

if(DBtime), dispVars( toc(time_start) ); end


%---------------------------------------------------------------------
% END; testing and plotting
%---------------------------------------------------------------------
if(DBtest)
    if(DBtime), time_start=tic; end
    tfd_test=full_gdtfd_testing_version(x,'swvd',dopp_win_params);
    if(DBtime), dispVars( toc(time_start) ); end
    if(Ntime==N)
        dispEE(tfd_test,tfd);
    else
        a=N/Ntime; 
        if( a==floor(a) )
            dispEE(tfd_test(1:a:end,:),tfd./a);
        end
    end
end
if(DBplot)
    figure(1); clf; 
    vtfd(tfd,real(x(1:N)));
    
    figure(9); clf; hold all;
    Z=fft(z);
    subplot(211); hold all; plot(sum(tfd)); plot( abs(Z(1:N)).^2./(2*N) );
    subplot(212); hold all; plot(sum(tfd)' - abs(Z(1:N)).^2./(2*N) );    
end




