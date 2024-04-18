# Change Log
All notable changes to this project will be documented in this file. Project started
2014-04-22 but this log started 2017-05-30, after version 0.2.3. More details can be found
in the git logs.


## unreleased
### Changed 
- implemented modified-B distribution in 'nonsep' section,
  e.g. tf=full_tfd(x,'nonsep',{'mb',0.01}); but is better to use lag-independent version,
  i.e. tf=full_tfd(x,'LI,{length(x)-1,'cosh',0.01});
### Removed
### Fixed
- input argument (Ntime) when calling LI kernel from full_tfd.m (was incorrectly passing
  Nfreq)
- bug fix in `gen_lag_kern.m` and `gen_Doppler_kern.m` for windows `gauss`, `cosh`, `tukey` as
default value was 0; now use Matlab will use default value
- missing `shiftWin` function (centres window) in `gen_lag_kern.m` and
`gen_Doppler_kern.m` (only used if 5 parameters in window cell); now added
### Added
- this CHANGELOG.md!
- windows: `blackmanharris`, `chebwin`, and `nuttall` (in `get_window.m`)

