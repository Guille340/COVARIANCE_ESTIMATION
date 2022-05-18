%  F = targetCovariance(ndof,DigitalFilter,varargin)
% 
%  DESCRIPTION
%  Returns the sample covariance matrix F generated from 10*NDOF normally 
%  distributed vectors of length NDOF simulated with function RANDN and 
%  filtered using the filter objects LOWPASSFILTER and DIGITALFILTER. The 
%  type of filtering (normal or zero-phase) is selected with FILTMODE.
%
%  DIGITALFILTER is a structure generated with DIGITALSINGLEFILTERDESIGN. 
%  LOWPASSFILTER is the vector of B coefficients of a lowpass filter, in 
%  particular, the frequency response of the antialias filter after resampling.
%
%  The covariance matrix F is used by function COVLOOC as the shrinkage target
%  for the estimation of the covariance matrix of a signal filtered with known 
%  filters DIGITALFILTER and LOWPASSFILTER. 
% 
%  INPUT ARGUMENTS (Fixed)
%  - ndof: number of degrees of freedom (i.e., variables or number of samples)
%    of the signal used to generate the covariance matrix. Therefore, F has
%    dimensions [NDOF NDOF].
%  - DigitalFilter: filter structure generated with DIGITALSINGLEFILTERDESIGN.
%
%  INPUT VARIABLES (Variable, Property/Value Pairs)
%  - 'LowPassFilter': b coefficients of the antialias/recovery filter use for 
%    resampling, after resampling (i.e., up to the new sampling frequency).
%  - 'FilterMode': filtering mode.
%    ¬ 'filter': standard filtering using FILTER function.
%    ¬ 'filtfilt': zero-phase filtering. The custom function MYFILTFILT is
%       used instead of FILTFILT (see function DIGITALSINGLEFILTER).
%  - 'DisplayProgress': TRUE for displaying the progress of TIMELIMIT. By 
%     default, DISPLAYPROGRESS = TRUE.
%    
%  OUTPUT VARIABLES
%  - F: target covariance matrix of dimensions [NDOF NDOF].
%
%  FUNCTION DEPENDENCIES
%  - digitalSingleFilter
%
%  FUNCTION CALL
%  1. F = targetCovariance(ndof,DigitalFilter)
%  2. F = targetCovariance(...,PROPERTY,VALUE)

%  VERSION 1.0
%  Date: 21 Nov 2021
%  Author: Guillermo Jimenez Arranz
%  Email: gjarranz@gmail.com

function F = targetCovariance(ndof,DigitalFilter,varargin)

% INPUT ARGUMENTS
% Verify number of Input Arguments
nFixArg = 2;
nProArg = 6;
narginchk(nFixArg,nFixArg + nProArg)
if rem(nargin - nFixArg,2)
    error('Variable input arguments must come in pairs (PROPERTY,VALUE)')
end

% Extract and Verify Input Properties
validProperties = {'lowpassfilter','filtermode','displayprogress'};
properties = lower(varargin(1:2:end));
if any(~ismember(properties,validProperties))
    error('One or more input properties are not recognised')
end

% Default Input Values
LowPassFilter = [];
filtMode = 'filter';
displayProgress = false;

% Extract and Verify Input Values
values = varargin(2:2:end);
nPairs = (nargin - nFixArg)/2; % number of (PROPERTY,VALUE) pairs
for m = 1:nPairs
    property = properties{m};
    switch property % populate with more properties if needed
        case 'lowpassfilter'
            LowPassFilter = values{m};
            if ~isempty(LowPassFilter) && (~isnumeric(LowPassFilter) ...
                    || ~isvector(LowPassFilter))
                LowPassFilter = [];
                warning(['LOWPASSFILTER is not a valid vector of B '...
                    'coefficients for an anti-alias FIR filter. '...
                    'No low-pass filtering will be applied in the '...
                    'calculation of the target covariance matrix'])
            end
        case 'filtermode'
            filtMode = values{m};
            if ~ischar(filtMode) || ~ismember(filtMode,{'filter','filtfilt'})
                filtMode = 'filter';
                warning(['Non-supported value for PROPERTY = '...
                    '''FilterMode''. ''FilterMode'' = %s will be used'],...
                    filtMode)
            end
        case 'displayprogress'
            displayProgress = values{m};
            if ~islogical(displayProgress) && ~any(displayProgress == [0 1])
                displayProgress = 0;
                warning(['Non-supported value for PROPERTY = '...
                    '''DisplayProgress''. A value of 0 will be used'])
            end
    end
end

% Error Control: NDOF
if ~isnumeric(ndof) || ~isscalar(ndof)
    error('NDOF must be a scalar number')
end

% Error Control: DIGITALFILTER
if ~isempty(DigitalFilter) && ~isDigitalSingleFilter(DigitalFilter)
    DigitalFilter = [];
     warning(['Input value for PROPERTY = ''DigitalFilter'' '...
        'is not a valid Digital Filter generated with '...
        'digitalSingleFilterDesign.m. No filtering will be '...
        'applied in the calculation of the target covariance matrix.'])
end 

% LPF Group Delay (in samples)
if ~isempty(LowPassFilter)
    gd = round(mean(grpdelay(LowPassFilter,100,LowPassFilter.SampleRate)));
end

% Display Progress (open)
if displayProgress
    h = waitbar(0,'','Name','covariance.m'); 
end

% SHRINKAGE TARGET MATRIX
nTests = 10*ndof; % number of tests needed for an accurate target matrix
minBlockSize = ndof*8/1024^2; % size of one row or test (MB)
maxBlockSize = max(50,minBlockSize); % maximum processing block size (MB)
nTestsPerBlock = min(floor(maxBlockSize*1024^2/(ndof*8)),nTests);
nBlocks = ceil(nTests/nTestsPerBlock); % number of processing blocks
nTests = nBlocks * nTestsPerBlock;
nFilters = numel(DigitalFilter);
F = zeros(ndof,ndof);
for k = 1:nBlocks
    % White Gaussian Vectors for Block
    X = randn(nTestsPerBlock,ndof);
    
    % Apply Filters
    for m = 1:nTestsPerBlock
        % Lowpass-filter the Observations
        if ~isempty(LowPassFilter)
            xf = filter(LowPassFilter,[X(m,:), zeros(1,gd)]); % zero-padded
            xf = xf(gd+1:end); % remove delayed segment
            X(m,:) = xf;
        end
        
        % Bandpass-filter the Observations
        for n = 1:nFilters
            X(m,:) = digitalSingleFilter(DigitalFilter(n),X(m,:),...
                'MetricsOutput',false,'FilterMode',filtMode,'DataWrap',true);
        end
        
        % Display Progress (if applicable)
        if displayProgress
            messageString = sprintf('Calculating shrinkage target (%d/%d)',...
                nTestsPerBlock*(k - 1) + m,nTests);
            waitbar((nTestsPerBlock*(k - 1) + m)/nTests,h,messageString);
        end
    end
    
    % X_mean = mean(X); % vector of means
    % X = X - X_mean; % matrix of deviation scores (ZERO MEAN ASSUMED!)
    F = (X'*X)/nTests + F; % target sample covariance matrix
end

F = F/mean(diag(F)); % filtered target covariance with unit mean variance

% Display Progress (close)
if displayProgress, close(h); end
