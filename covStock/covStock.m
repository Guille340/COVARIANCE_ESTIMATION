%  [C,shrink] = covStock(X,varargin)
% 
%  DESCRIPTION
%  Computes the covariance matrix from a matrix of raw scores X using the 
%  Ledoit & Wolf (2001) method.
%
%  The shrinkage method is suitable for high-dimensional problems (large number
%  of variables N) with a small number of samples (few observations M). The 
%  shrinkage approach attempts to address the low accuracy of the sample 
%  covariance matrix (X'*X/M) when M < N by introducing bias. 
%
%  COVSTOCK returns the shrunk covariance matrix C and shrinkage coefficient 
%  SHRINK. The shrunk covariance C is estimated as a linear combination of
%  the sample covariance matrix S and a shrinkage target matrix F by means of 
%  the shrinkage coefficient SHRINK. C computed as follows:
% 
%       C = (1-RHO)*S + RHO*F
%
%  In this case, the target matrix F is the covariance matrix of stock returns.
% 
%  INPUT ARGUMENTS (Fixed)
%  - X: matrix of raw scores with as many rows as observations (M) and as many
%    columns as variables (N).
%
%  INPUT ARGUMENTS (Variable, Property/Value Pairs)
%  - 'ShrinkCoef': shrinkage coefficient or "intensity". If set as empty ([]), 
%     the shrinkage coefficient is calculated automatically. By default, 
%     SHRINKCOEF = [].
%    
%  OUTPUT VARIABLES
%  - C: covariance estimate.
%  - shrink: shrinkage coefficient.
%
%  FUNCTION DEPENDENCIES
%  - None
%
%  FUNCTION CALL
%  1. [C,shrink] = covStock(X)
%  2. [C,shrink] = covStock(...,PROPERTY,VALUE)
%       ¬ 'ShrinkCoef'
%
%  REFERENCES
%  - Ledoit, O., and Wolf, M. (2001). "Improved estimation of the covariance
%    matrix of stock returns with an application to portfolio selection".
%  - https://www.econ.uzh.ch/en/people/faculty/wolf/publications.html
%  - "Estimation of Covariance Matrices", Wikipedia article.
%    https://en.wikipedia.org/wiki/Estimation_of_covariance_matrices

%  VERSION 1.0
%  Date: 28 Oct 2021
%  Author: Guillermo Jimenez Arranz
%  Email: gjarranz@gmail.com

function [C,shrink] = covStock(X,varargin)

% INPUT ARGUMENTS
% Verify number of Input Arguments
nFixArg = 1;
nProArg = 2;
narginchk(nFixArg,nFixArg + nProArg)

% Verify Number of Property/Value Pairs
if rem(nargin - nFixArg,2)
    error('Variable input arguments must come in pairs (PROPERTY,VALUE)')
end

% Extract and Verify Input Properties
validProperties = {'shrinkcoef'};
properties = lower(varargin(1:2:end));
if any(~ismember(properties,validProperties))
    error('One or more input properties are not recognised')
end

% Default Input Values
shrink = [];

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
    end
end

% Error Control: X
if ~isnumeric(X) || ~ismatrix(X)
    error('X must be a numeric matrix')
end

% MATRIX OF DEVIATION SCORES
[M,N] = size(X);
X_mean = mean(X); % vector of means
X = X - X_mean; % deviation scores matrix

% SHRINKAGE TARGET
F = 0; % initialise target covariance matrix
if isempty(shrink) || shrink ~= 0
    % Variance and Covariance of Market Returns
    X_mkt = mean(X,2);
    S = ([X X_mkt]'*[X X_mkt])/M; % extended sample covariance matrix
    var_mkt = S(N+1,N+1); % variance of market returns
    cov_mkt = S(1:N,N+1); % covariance of market returns

    % Sample Covariance
    S(:,N+1) = [];
    S(N+1,:) = [];

    % Shrinkage Target
    F = cov_mkt*cov_mkt'/var_mkt;
    iDiag = (N+1)*(0:N-1) + 1;
    F(iDiag) = diag(S);
else
    S = (X'*X)/M; % sample covariance matrix 
end

% SHRINKAGE COEFFICIENT
if isempty(shrink)
    % Compute "c" 
    c = norm(S-F,'fro')^2;

    % Compute "p"
    X2 = X.^2; % squared deviation scores
    p_mat = X2'*X2/M - S.^2; % pi matrix
    p_diag = diag(p_mat);
    p = sum(p_mat,'all'); % pi-hat
    clear p_mat

    % Compute "r" (diagonal)
    r_diag = sum(p_diag); % first sum term

    % Compute "r" (off-diagonal)
    Z = X.*X_mkt;
    v1 = X2'*Z/M - cov_mkt.*S;
    v3 = Z'*Z/M - var_mkt*S;
    r_off1 = sum(v1.*cov_mkt'/var_mkt,'all') ...
        - sum(diag(v1).*cov_mkt/var_mkt);
    r_off3 = sum(v3.*(cov_mkt*cov_mkt')/var_mkt^2,'all') ...
        - sum(diag(v3).*cov_mkt.^2/var_mkt^2);
    r_off = 2*r_off1 - r_off3;

    % Compute "r" (total)
    r = r_diag + r_off;

    % Compute Shrinkage
    k = (p-r)/c;
    shrink = max(0,min(1,k/M));
end

% SHRINKED COVARIANCE
C = (1-shrink)*S + shrink*F;
