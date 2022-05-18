%  [C,shrink] = covLooc(X,varargin)
% 
%  DESCRIPTION
%  Computes the covariance matrix from a matrix of raw scores X using the
%  shrinkage method with cross-validation described in Theiler (2012).
%
%  The shrinkage method is suitable for high-dimensional problems (large number
%  of variables N) with a small number of samples (few observations M). The 
%  shrinkage approach attempts to address the low accuracy of the sample 
%  covariance matrix (X'*X/M) when M < N by introducing bias. 
%
%  COVLOOC returns the shrunk covariance matrix C and shrinkage coefficient 
%  SHRINK. The shrunk covariance C is estimated as a linear combination of
%  the sample covariance matrix S and a shrinkage target matrix F by means of 
%  the shrinkage coefficient SHRINK. C computed as follows:
% 
%       C = (1-RHO)*S + RHO*F
%
%  The target matrix F can be specified as input with property 'ShrinkTarget'.
%  By default, F is the diagonal matrix with entries equal to those in the
%  sample covariance matrix.
% 
%  INPUT ARGUMENTS (Fixed)
%  - X: matrix of raw scores with as many rows as observations (M) and as many
%    columns as variables (N).
%
%  INPUT ARGUMENTS (Variable, Property/Value Pairs)
%  - 'ShrinkCoef': shrinkage coefficient or "intensity". If set as empty ([]), 
%     the shrinkage coefficient is calculated automatically. By default, 
%     SHRINKCOEF = [].
%  - 'ShrinkTarget': shinkage target. By default, SHRINKTARGET = DIAG(S).
%  - 'ShrinkMethod': method for finding the optimal shrinkage coefficient.
%    ¬ 'optim': uses FMINSEARCH to find the SHRINK that minimises the mean
%      negative log-likelihood.
%    ¬ 'interp': computes the mean negative-log likelihood from SHRINK = 1e-12
%      to SHRINK = 1 in geometric steps (1.259 spearation factor), to then 
%      obtain the value of SHRINK associated with the minimum MNLL.
%  - 'DisplayProgress': TRUE for displaying the progress of TIMELIMIT. By 
%     default, DISPLAYPROGRESS = TRUE.
%    
%  OUTPUT VARIABLES
%  - C: covariance estimate.
%  - shrink: shrinkage coefficient.
%
%  FUNCTION DEPENDENCIES
%  - looc
%
%  FUNCTION CALL
%  1. [C,shrink] = covLooc(X)
%  2. [C,shrink] = covLooc(...,PROPERTY,VALUE)
%       ¬ 'ShrinkCoef', 'ShrinkTarget', 'DisplayProgress'
%
%  REFERENCES
%  - Theiler, J. (2012). "The incredible shrinking covariance estimator", 
%    Proc. SPIE 8391.
%  - Hoofbeck, J. P., and Landgrebe, D. A. (1996). "Covariance matrix
%    estimation and classification with limited training samples", IEEE
%    Trans. Pattern Analysis and Machine Intelligence, 18, p. 763-767.
%  - Burg, J. P., Luenberger, D. G., and Wenger, D. L. (1982). "Estimation
%    of structured covariance matrices", Proceedings of the IEEE, 70, p.
%    963-974.
%  - "Estimation of Covariance Matrices", Wikipedia article.
%    https://en.wikipedia.org/wiki/Estimation_of_covariance_matrices

%  VERSION 1.0
%  Date: 28 Oct 2021
%  Author: Guillermo Jimenez Arranz
%  Email: gjarranz@gmail.com

function [C,shrink] = covLooc(X,varargin)

% INPUT ARGUMENTS
% Verify number of Input Arguments
nFixArg = 1;
nProArg = 8;
narginchk(nFixArg,nFixArg + nProArg)

% Error Control: X
if ~isnumeric(X) || ~ismatrix(X)
    error('X must be a numeric matrix')
end
[M,N] = size(X);

% Verify Number of Property/Value Pairs
if rem(nargin - nFixArg,2)
    error('Variable input arguments must come in pairs (PROPERTY,VALUE)')
end

% Extract and Verify Input Properties
validProperties = {'shrinkcoef','shrinktarget','shrinkmethod',...
    'displayprogress'};
properties = lower(varargin(1:2:end));
if any(~ismember(properties,validProperties))
    error('One or more input properties are not recognised')
end

% Default Input Values
shrink = [];
F = 'origdiag';
shrinkMethod = 'interp';
displayProgress = false;

% Extract and Verify Input Values
values = varargin(2:2:end);
nPairs = (nargin - nFixArg)/2; % number of (PROPERTY,VALUE) pairs
for m = 1:nPairs
    property = properties{m};
    switch property % populate with more properties if needed
        
        case 'shrinkcoef'
            shrink = values{m};
            if ~isempty(shrink) && (~isnumeric(shrink) || ~isscalar(shrink) ...
                    || shrink < 0 || shrink > 1)
                shrink = [];
                warning(['Non-valid value for PROPERTY = ''ShrinkCoef''. '...
                    'SHRINKCOEF must be a numeric scalar between 0 and 1. '...
                    'SHRINKCOEF will be calculated automatically.'],shrink)
            end
            
        case 'shrinktarget'
            F = values{m};
            if ischar(F)
                if ~ismember(F,{'origdiag','meandiag'})
                    F = 'origdiag';
                    warning(['F is not a supported character vector for '...
                        'a target covariance matrix. F = %s will be used'],F)
                end
            elseif isnumeric(F) && ismatrix(F)
                [M0,N0] = size(F);
                if M0 ~= N0 || N0 ~= N
                    F = 'origdiag';
                    warning(['Non-valid value for PROPERTY = ''ShrinkTarget''.'...
                        'SHRINKTARGET must be a valid target covariance '...
                        'matrix (square with as many variables as X). '...
                        'SHRINKTARGET = %s will be used'],F)     
                end
            else
                F = 'origdiag';
                warning(['Non-valid value for PROPERTY = ''ShrinkTarget''. '...
                    'SHRINKTARGET must be a valid character string '...
                    '(''origdiag'', ''meandiag'') or a square matrix with '...
                    'as many elements as variables are in the raw scores '...
                    'matrix X. SHRINKTARGET = %s will be used'],F)
            end 
        case 'shrinkmethod'
            shrinkMethod = values{m};
            if ~ischar(shrinkMethod) || ~ismember(shrinkMethod,...
                    {'optim','interp'})
                    shrinkMethod = 'interp';
                    warning(['Non-valild value for PROPERTY = '...
                        '''ShrinkMethod''. SHRINKMETHOD must be a valid '...
                        'character string (''optim'', ''interp''). '...
                        'SHRINKMETHOD = %s will be used'],shrinkMethod)
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

% Display Progress (open)
if displayProgress
    h = waitbar(0,'','Name','covariance.m'); 
end

% SAMPLE COVARIANCE
% X_mean = mean(X); % vector of means
% X = X - X_mean; % deviation scores matrix (ASSUMED ZERO MEAN!)
S = (X'*X)/M; % sample covariance matrix

% SHRINKAGE TARGET
if ischar(F)
    if isempty(shrink) || shrink ~= 0
        if strcmp(F,'origdiag')
            F = diag(diag(S));
        else % F = 'meandiag'
            F = mean(diag(S)) * eye(size(S));
        end
    else
        F = 0;
    end
else % square matrix
    F = F*mean(diag(S))/mean(diag(F));
end
    
% SHRINKAGE COEFFICIENT
if isempty(shrink)
    if strcmp(shrinkMethod,'optim')
        % Optimisation Method
        fun = @(alphaLog) looc(alphaLog,X,S,F,'hl','log');
        options = optimset('Display','iter','TolX',0.1,'MaxIter',100);
        alphaLog = fminsearch(fun,-0.01,options);
        shrink = max(min(10^alphaLog,1),0); % limit shrink between 0 and 1
    else
        % Interpolation Method
        alpha = 10.^(-12:0.1:0); % alphalog
        nAlpha = length(alpha);
        nll = zeros(nAlpha,1);
        for m = 1:nAlpha
            % Compute Non-Negative Likelihood for Shrinkage Coefficient
            nll(m) = looc(alpha(m),X,S,F,'hl','lin'); % negative-log likelihood

            % Display Progress (if applicable)
            if displayProgress
                messageString = sprintf('Calculating shrinkage coefficient (%d/%d)',...
                    m,nAlpha);
                waitbar(m/nAlpha,h,messageString);
            end
        end

        % Find Alpha Values for the Smallest NLL 5% (better fit)
        [~,inll_sort] = sort(nll,'ascend');
        alpha_sort = alpha(inll_sort);
        i2 = max(min(round(0.05*nAlpha),nAlpha),1);
        alpha_5per = alpha_sort(1:i2);
        alpha_5per = alpha_5per(~isinf(alpha_5per)); % remove Inf and -Inf values
        shrink = max(alpha_5per);

        % Find Minimum Alpha (worse fit)
        % [~,iMin] = min(nll);
        % shrink = alpha(iMin);
    end
end

% SHRINKED COVARIANCE
C = (1-shrink)*S + shrink*F; % regularised sample covariance matrix

% Display Progress (close)
if displayProgress, close(h); end

% PLOT SHRINKAGE OPTIMISATION (PROVISIONAL)
% figure
% plot(alpha,nll,'b')
% xlabel('alpha')
% ylabel('NLL')
% title(sprintf('SHRINK = %0.1e',shrink))
% set(gca,'XScale','log')
