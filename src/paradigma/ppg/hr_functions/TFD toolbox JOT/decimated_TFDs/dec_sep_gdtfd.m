%-------------------------------------------------------------------------------
% dec_sep_gdtfd: Decimated TFD with separable kernel of the form g[l,m]=G1[l]g2[m]. For
% efficient implement define windows G1[l] and g2[m] as band-limited functions.
%
% TFD is decimated with integer factors a,b as ρ[an,bk]
%
% Syntax: tfd=dec_sep_gdtfd(x,dopp_win_params,lag_win_params,time_dec,freq_dec,Ntime,Nfreq)
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
%                     domain, which is the default) or 0 (define window in the time domain)
%                 e.g. {121, 'hamm'}; {121, 'tukey', 0.2}; {127,'cosh',0.01,0}
%
%      lag_win_params = lag window parameters in cell form:
%                 {win_length,win_type,win_param,lag_or_not} where
%                     - win_length is the sample length of the window
%                     - win_type is the type of window 
%                     - [optional] win_param is the parameter of the window 
%                     - [optional] lag_or_not is either 0 (define window in the lag
%                     domain, which is the default) or 1 (define window the frequency domain)
%                 e.g. {121, 'hamm'}; {121, 'tukey', 0.2}; {127,'cosh',0.01,0}
%
%      time_dec  = decimation factor a in the time domain; a/Ntime is integer
%      freq_dec  = decimation factor b in the frequency domain; b/Nfreq is integer
%
%      Nfreq = frequency oversampling value; must be greater than length of lag window
%      Ntime = time oversampling value; must be greater than length of Doppler window
%
% Outputs: 
%     tfd = (a/Ntime) x (b/Nfreq) time–frequency distribution
%
% See also: SEP_GDTFD, GET_ANALYTIC_SIGNAL, GEN_LAG_KERN, GEN_DOPPLER_KERN, FFT
%
% Example:
%     N=1024; Ntime=64; Nfreq=128; a=1; b=2;
%     x=gen_LFM(N,0.1,0.3)+gen_LFM(N,0.4,0.1);
%
%     c=dec_sep_gdtfd(x,{51,'hann'},{101,'hann'},a,b,Ntime,Nfreq);
%     vtfd(c,x);

% John M. O' Toole, University College Cork
% Started: 24-04-2014
%
% last update: Time-stamp: <2014-07-23 15:25:51 (otoolej)>
%-------------------------------------------------------------------------------
function tfd=dec_sep_gdtfd(x,dopp_win_params,lag_win_params,time_dec,freq_dec,Ntime, ...
                           Nfreq)
if(nargin<2 || isempty(dopp_win_params)), dopp_win_params={51,'hann'}; end
if(nargin<3 || isempty(lag_win_params)), lag_win_params={151,'hann'}; end
if(nargin<4 || isempty(time_dec)), time_dec=1; end
if(nargin<5 || isempty(freq_dec)), freq_dec=1; end
if(nargin<6 || isempty(Ntime)), Ntime=[]; end
if(nargin<7 || isempty(Nfreq)), Nfreq=[]; end

DBplot=0;
DBmem=0;
DBtest=0;
DBtime=0;
DBverbose=0;


if(DBtime), time_start=tic; end
%---------------------------------------------------------------------
% 1. convert real-valued signal to analytic signal of length 2N
%---------------------------------------------------------------------
[z,N2,N,Nh]=get_analytic_signal(x);


%---------------------------------------------------------------------
% 2. generate kernel windows and check decimation parameters
%---------------------------------------------------------------------
[G1,Q,Ntime]=gen_Doppler_kern(dopp_win_params,N,Ntime);
Qh=ceil(Q/2);

[g2,P,Ph_floor,Nfreq]=gen_lag_kern(lag_win_params,N,Nfreq);
Nh=ceil(Nfreq/2); Ph=ceil(P/2);


% check decimation parameters and return as sequence:
if(length(time_dec)>1 || length(freq_dec)>1)
    error('Frequency and time decimation parameters should be scalar');
end
[n_seq,L,time_dec]=check_dec_params_seq(time_dec,Ntime,'time',1);
[k_seq,J,freq_dec]=check_dec_params_seq(freq_dec,Nfreq,'frequency',1);
Jh=ceil(J/2);  J_extend=2*Jh+2; P_extend=2*Ph+2;

if(DBmem), s=whos; fprintf('start: mem=%s\n',disp_bytes(sum([s.bytes]))); end

tfd=zeros(L,J_extend); 
m_real=1:(Jh+1); m_imag=(J_extend-Jh):(J_extend);

if(DBverbose), 
    dispVars(N,Ntime,Nfreq,L,J,Jh,J_extend,P,Ph,Q,Qh,P_extend); 
    dispVars(length(m_real),length(m_imag),m_real(end),m_imag(1));
end

if(DBmem), s=whos; fprintf('declare TFD: mem=%s\n',disp_bytes(sum([s.bytes]))); end


%---------------------------------------------------------------------
% 2. generate time--lag signal function (for positive-lag values only)
%---------------------------------------------------------------------
l=0:(Qh-1); lb=1:(Qh-1);
n=0:N-1;
for m=0:Jh

    %-------------------------------------------------------------------------
    % a) Form lag-slice of windowed time-lag function and then fold in
    %    the lag direction.
    %-------------------------------------------------------------------------
    R_lag_slice=zeros(N,1);
    for p=0:freq_dec-1
        mmod=p.*J+m;
        
        if(mmod<=Ph)
            inp=mod(n+mmod,N2);  inn=mod(n-mmod,N2);           
            i1=mod(n+mmod,N2); i2=mod(n-mmod,N2);

            R_lag_slice=R_lag_slice+z(inp+1).*conj(z(inn+1)).*g2(mmod+1);
        elseif(mmod>Nfreq-Ph)
            inp=mod(n+Nfreq-mmod,N2); inn=mod(n-Nfreq+mmod,N2);
            
            R_lag_slice=R_lag_slice+conj(z(inp+1)).*z(inn+1).*g2(P-Nfreq+mmod+1);
        end
    end
    

    %-------------------------------------------------------------------------
    % b) DFT to the Doppler--lag domain..
    %-------------------------------------------------------------------------
    R_lag_slice=fft(R_lag_slice);

    %-------------------------------------------------------------------------
    % c) Multiply by Doppler window G1
    %-------------------------------------------------------------------------
    r_lag_slice=zeros(Ntime,1);
    r_lag_slice(l+1)=R_lag_slice(l+1).*G1(l+1);
    r_lag_slice(Ntime-lb+1)=R_lag_slice(N-lb+1).*G1(Q-lb+1);    
    
    %-------------------------------------------------------------------------
    % d) Fold in the Doppler direction
    %-------------------------------------------------------------------------
    r_lag_fold=fold_vector_full(r_lag_slice,L,time_dec);
  
    %-------------------------------------------------------------------------
    % e) DFT the lag slice to the time--lag domain
    %-------------------------------------------------------------------------
    r_lag_fold=ifft(r_lag_fold);

    tfd(:,m_real(m+1))=real(r_lag_fold);
    tfd(:,m_imag(m+1))=imag(r_lag_fold);    
end


if(DBmem), s=whos; fprintf('R: mem=%s\n',disp_bytes(sum([s.bytes]))); end

%-------------------------------------------------------------------------
% 4.  Expand R for positive and negative lag values and DFT back to 
%     time--frequency domain
%-------------------------------------------------------------------------
m=0:Jh; mb=1:(Jh-1);
for n=0:2:(L-2)
    R_even_half=complex(tfd(n+1,m_real),tfd(n+1,m_imag));
    R_odd_half =complex(tfd(n+2,m_real),tfd(n+2,m_imag));
    
    R_tslice_even=zeros(J,1);  R_tslice_odd=zeros(J,1);
    R_tslice_even(m+1)=R_even_half(m+1);
    R_tslice_odd(m+1) =R_odd_half(m+1);
    R_tslice_even(J-mb+1)=conj( R_even_half(mb+1) );
    R_tslice_odd(J-mb+1) =conj( R_odd_half(mb+1) );
    
    tfd_time_slice=fft( R_tslice_even+j.*R_tslice_odd );

    tfd(n+1,1:J)=real(tfd_time_slice);
    tfd(n+2,1:J)=imag(tfd_time_slice);
end

% one extra FFT if L is odd:
if(rem(L,2))
    R_even_half=complex(tfd(L,m_real),tfd(L,m_imag));
    
    R_tslice_even=zeros(J,1);  
    R_tslice_even(m+1)=R_even_half(m+1);
    R_tslice_even(J-mb+1)=conj( R_even_half(mb+1) );
    
    tfd_time_slice=fft( R_tslice_even+j.*R_tslice_odd );

    tfd(L,1:J)=real(tfd_time_slice);
end

tfd=tfd(:,1:J);

if(DBmem), s=whos; fprintf('end: mem=%s\n',disp_bytes(sum([s.bytes]))); end

scale_factor=1/Nfreq;
tfd=tfd.*scale_factor;

if(DBtime), dispVars( toc(time_start) ); end

%---------------------------------------------------------------------
% END; testing and plotting
%---------------------------------------------------------------------
if(DBtest)
    if(DBtime),  time_start=tic; end
    tfd_test=full_gdtfd_testing_version(x,'sep',{dopp_win_params,lag_win_params});
    if(DBtime),  dispVars( toc(time_start) ); end

    a=N/Ntime; b=N/Nfreq; 
    if( a==floor(a) && b==floor(b) )
        dispEE(tfd_test( (a.*n_seq)+1,(b.*k_seq)+1),tfd./(a*b));
    end
    
% $$$     figure(100); clf; hold all;
% $$$     k=10; plot(tfd_test(n_seq+1,k_seq(k)+1)); plot(tfd(:,k+1));
% $$$     keyboard;
end
if(DBverbose)
    fprintf('size=%dx%d; max=%g; total energ:%d\n', size(tfd,1), size(tfd,2), max(tfd(:)), ...
            sum(tfd(:)));
end
if(DBplot)
    figure(1); clf; 
    vtfd(tfd,real(x(1:N)));
    
% $$$     figure(9); clf; hold all;
% $$$     subplot(211); hold all; plot(sum(tfd')'); plot( abs(z(1:N)).^2 );
% $$$     subplot(212); hold all; plot(sum(tfd')' - abs(z(n_seq+1)).^2 );    
% $$$     
% $$$     figure(10); clf; hold all;
% $$$     Z=fft(z);
% $$$     subplot(211); hold all; plot(sum(tfd)); plot( abs(Z(1:N)).^2./(2*N) );
% $$$     subplot(212); hold all; plot(sum(tfd)' - abs(Z(k_seq+1)).^2./(2*N) );    
end
