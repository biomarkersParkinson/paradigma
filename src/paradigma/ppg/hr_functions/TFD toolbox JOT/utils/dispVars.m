%-------------------------------------------------------------------------------
% Display variables to standard output
%
% DATE: 07-09-2010
%-------------------------------------------------------------------------------
function dispVars( varargin )
% $$$ for i=1:nargin
% $$$     keyboard;
% $$$     varargin{i}
% $$$ end

str = '';
for i=1:nargin
  if( iscell(varargin{i}) )
    for j=1:length(varargin{i})
        str = [ str ' |' char(inputname(i)) '(' num2str(j) ') =' ...
                num2str(varargin{i}{j}) ];
    end
  else
    
    str = strcat(str,' |',char(inputname(i)),'=',num2str(varargin{i}));
  end
  
end

disp(str);
