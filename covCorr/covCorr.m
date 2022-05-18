%  [C,shrink] = covCorr(X,varargin)
% 
%  DESCRIPTION
%  Computes the covariance matrix from a matrix of raw scores X using the 
%  Ledoit & Wolf (2003) method.
%
%  The shrinkage method is suitable for high-dimensional problems (large number
%  of variables N) with a small number of samples (few observations M). The 
%  shrinkage approach attempts to address the low accuracy of the sample 
%  covariance matrix (X'*X/M) when M < N by introducing bias. 
%
%  COVCORR returns the shrunk covariance matrix C and shrinkage coefficient 
%  SHRINK. The shrunk covariance C is estimated as a linear combination of
%  the sample covariance matrix S and a shrinkage target matrix F by means of 
%  the shrinkage coefficient SHRINK. C computed as follows:
% 
%       C = (1-RHO)*S + RHO*F
%
%  In this case, the target matrix F is a form of correlation matrix with
%  the diagonal entries being identical to those in the sample covariance 
%  matrix.
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
%  1. [C,shrink] = covCorr(X)
%  2. [C,shrink] = covCorr(...,PROPERTY,VALUE)
%       ¬ 'ShrinkCoef'
%
%  REFERENCES
%  - Ledoit, O., and Wolf, M. (2003). "Honey, I shrunk the sample covariance
%    matrix".
%  - https://www.econ.uzh.ch/en/people/faculty/wolf/publications.html
%  - "Estimation of Covariance Matrices", Wikipedia article.
%    https://en.wikipedia.org/wiki/Estimation_of_covariance_matrices

%  VERSION 1.0
%  Date: 28 Oct 2021
%  Author: Guillermo Jimenez Arranz
%  Email: gjarranz@gmail.com

function [C,shrink] = covCorr(X,varargin)

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

% SAMPLE COVARIANCE
[M,N] = size(X);
X_mean = mean(X); % vector of means
X = X - X_mean; % deviation scores matrix
S = (X'*X)/M; % sample covariance matrix

% SHRINKAGE TARGET
F = 0; % initialise target covariance matrix
if isempty(shrink) || shrink ~= 0
    S_var = diag(S);
    S_std = sqrt(S_var);
    rmean = (sum(S./(S_std.*S_std'),'all') - N)/(N*(N-1));
    F = rmean * S_std.*S_std';
    iDiag = (N+1)*(0:N-1) + 1;
    F(iDiag) = S_var;
    F = (F + F')/2; % make matrix strictly symmetric
end

% SHRINKAGE COEFFICIENT
if isempty(shrink)
    % Compute pi-hat
    X2 = X.^2; % squared deviation scores
    pi_mat = X2'*X2/M - S.^2; % pi matrix
    pi_diag = diag(pi_mat);
    pi_hat = sum(pi_mat,'all'); % pi-hat
    clear X2 pi_mat

    % Compute rho-hat
    theta_mat = (X.^3)'*X/M - S_var.*S; % theta matrix
    theta_mat(iDiag) = zeros(N,1); % set diagonal to zero
    sum1 = sum(pi_diag); % first sum term
    sum2 = rmean*sum(((1./S_std)*S_std').*theta_mat,'all'); % second sum term
    rho_hat = sum1 + sum2;

    % Compute gamma-hat
    gamma_hat = norm(S - F,'fro')^2;

    % Compute Shrinkage
    kappa_hat = (pi_hat - rho_hat)/gamma_hat;
    shrink = max(0,min(1,kappa_hat/M));
end

% SHRINKED COVARIANCE
C = (1-shrink)*S + shrink*F; % regularised sample covariance matrix
