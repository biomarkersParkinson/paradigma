%-------------------------------------------------------------------------------
% dec_di_gdtfd: decimated Doppler-independent kernel TFD ρ[nᵢ,bk], where b is an
% integer value and nᵢ is the set nᵢ ={ nᵢ | 1≤i≤U }, with 0≤nᵢ≤N-1
%
% Syntax: tfd=dec_di_gdtfd(x,lag_win_params,time_dec,freq_dec,Nfreq)
%
% Inputs: 
%      x = input signal (either real-valued signal of length-N or
%          complex-valued analytic signal of length-2N)
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
%      time_dec = decimation factor a in the time domain; a/Ntime is integer
%      freq_dec = decimation factor b in the frequency domain; b/Nfreq is integer
%
%      Nfreq = frequency oversampling value; must be greater than length of lag window
%
% See also: DI_GDTFD, GET_ANALYTIC_SIGNAL, GEN_LAG_KERN, FFT
%
% Outputs: 
%     tfd = U x (b/Nfreq) time–frequency distribution
%
% Example:
%     N=1024; Nfreq=128; ni=[100:4:900]; b=2;
%     x=gen_LFM(N,0.1,0.3)+gen_LFM(N,0.4,0.1);
%     
%     c=dec_di_gdtfd(x,{101,'hann'},ni,b,Nfreq);
%     vtfd(c,x,1,ni);


% John M. O' Toole, University College Cork
% Started: 23-04-2014
%
% last update: Time-stamp: <2014-07-23 16:22:59 (otoolej)>
%-------------------------------------------------------------------------------
function tfd=dec_di_gdtfd(x,lag_win_params,time_dec,freq_dec,Nfreq)
if(nargin<2 || isempty(lag_win_params)), lag_win_params={101,'hamm'}; end
if(nargin<3 || isempty(time_dec)), time_dec=[]; end
if(nargin<4 || isempty(freq_dec)), freq_dec=1; end
if(nargin<5 || isempty(Nfreq)), Nfreq=[]; end

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
% 2. generate time--lag signal function (for positive-lag values only)
%---------------------------------------------------------------------
[g2,P,Ph_floor,Nfreq]=gen_lag_kern(lag_win_params,N,Nfreq);
Nh=ceil(Nfreq/2);
Ph=ceil(P/2);


% check decimation parameters and return as sequence:
if(length(freq_dec)>1)
    error('Frequency decimation parameter should be scalar');
end
[n_seq,U]=check_dec_params_seq(time_dec,N,'time',0);
[k_seq,J,freq_dec]=check_dec_params_seq(freq_dec,Nfreq,'frequency',1);
Jh=ceil(J/2);  J_extend=2*Jh+2;


if(DBmem), s=whos; fprintf('start: mem=%s\n',disp_bytes(sum([s.bytes]))); end
tfd=zeros(U,J_extend); 
if(J~=Nfreq)
    m_real=1:Jh+1; m_imag=(J_extend-Jh):(J_extend);
else
    m_real=1:Ph; m_imag=(J-Ph+1):J;
end



m=0:(Ph-1); mb=1:(Ph-1);
for in=1:U
    n=n_seq(in);
    inp=mod(n+m,N2);  inn=mod(n-m,N2); 
    
    if(J~=Nfreq)
        % need to fold signal function (for decimation in frequency-direction):        
        K_time_slice=zeros(1,Nfreq);
        K_time_slice(m+1)=g2(m+1).*z(inp+1).*conj( z(inn+1) );
        K_time_slice(Nfreq-mb+1)=conj(K_time_slice(mb+1));

        R_fold=fold_vector_half(K_time_slice,J,Jh,freq_dec);
    else
        % or if not then do as usual:
        K_time_slice(m+1)=g2(m+1).*z(inp+1).*conj( z(inn+1) );

        R_fold=K_time_slice;
    end
    
    tfd(in,m_real)=real( R_fold );
    tfd(in,m_imag)=imag( R_fold );    
end

if(DBmem), s=whos; fprintf('K: mem=%s\n',disp_bytes(sum([s.bytes]))); end


%-------------------------------------------------------------------------
% 3.  Expand R for positive and negative lag values and DFT back to 
%     time--frequency domain
%-------------------------------------------------------------------------
if(J~=Nfreq)
    m=0:Jh; mb=1:(Jh-1);
else
    m=0:(Ph-1); mb=1:(Ph-1);
end
    

for n=0:2:(U-2)
    R_even_half=complex(tfd(n+1,m_real),tfd(n+1,m_imag));
    R_odd_half =complex(tfd(n+2,m_real),tfd(n+2,m_imag));    
    
    R_tslice_even=zeros(1,J);  R_tslice_odd=zeros(1,J);
    R_tslice_even(m+1)=R_even_half(m+1);
    R_tslice_odd(m+1) =R_odd_half(m+1);
    R_tslice_even(J-mb+1)=conj( R_even_half(mb+1) );
    R_tslice_odd(J-mb+1) =conj( R_odd_half(mb+1) );
    
    tfd_time_slice=fft( R_tslice_even+j.*R_tslice_odd );

    tfd(n+1,1:J)=real(tfd_time_slice);
    tfd(n+2,1:J)=imag(tfd_time_slice);
end

% one extra FFT if U is odd:
if(rem(U,2))
    R_even_half=complex(tfd(U,m_real),tfd(U,m_imag));
    R_tslice_even=zeros(1,J);
    R_tslice_even(m+1)=R_even_half(m+1);
    R_tslice_even(J-mb+1)=conj( R_even_half(mb+1) );
    tfd_time_slice=fft( R_tslice_even );

    tfd(U,1:J)=real(tfd_time_slice);
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
    tfd_test=full_gdtfd_testing_version(x,'pwvd',{P,lag_win_params{2:end}});
    if(DBtime),  dispVars( toc(time_start) ); end

    if(Nfreq==N)
        dispEE(tfd_test(n_seq+1,k_seq+1),tfd);        
    else
        b=N/Nfreq; 
        if( b==floor(b) )
            dispEE(tfd_test(n_seq+1,(b.*k_seq)+1),tfd./b);
        end
    end

end
if(DBverbose)
    fprintf('size=%dx%d; max=%g; total energ:%d\n', size(tfd,1), size(tfd,2), max(tfd(:)), ...
            sum(tfd(:)));
end
if(DBplot)
    figure(1); clf; 
    vtfd(tfd,real(x(1:N)));
    
    figure(9); clf; hold all;
    subplot(211); hold all; plot(sum(tfd')'); plot( abs(z(1:N)).^2 );
    subplot(212); hold all; plot(sum(tfd')' - abs(z(n_seq+1)).^2 );    
end


