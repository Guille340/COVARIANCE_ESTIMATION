%  [C,shrink] = covDiag(X,varargin)
% 
%  DESCRIPTION
%  Computes the covariance matrix from a matrix of raw scores X using the 
%  Ledoit method with diagonal shrinkage target.
%
%  This shrinkage method is suitable for high-dimensional problems (large 
%  number of variables N) with a small number of samples (few observations M). 
%  The shrinkage approach attempts to address the low accuracy of the sample 
%  covariance matrix (X'*X/M) when M < N by introducing bias. 
%
%  COVDIAG returns the shrunk covariance matrix C and shrinkage coefficient 
%  SHRINK. The shrunk covariance C is estimated as a linear combination of
%  the sample covariance matrix S and a shrinkage target matrix F by means of 
%  the shrinkage coefficient SHRINK. C computed as follows:
% 
%       C = (1-RHO)*S + RHO*F
%
%  In this case, the target matrix F is the diagonal matrix with idnetical
%  entries to those in the sample covariance matrix.
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
%  1. [C,shrink] = covDiag(X)
%  2. [C,shrink] = covDiag(...,PROPERTY,VALUE)
%       ¬ 'ShrinkCoef'
%
%  REFERENCES
%  - http://ledoit.net/shrinkDiag.m
%  - "Estimation of Covariance Matrices", Wikipedia article.
%    https://en.wikipedia.org/wiki/Estimation_of_covariance_matrices

%  VERSION 1.0
%  Date: 28 Oct 2021
%  Author: Guillermo Jimenez Arranz
%  Email: gjarranz@gmail.com

function [C,shrink] = covDiag(X,varargin)

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
M = size(X,1);
X_mean = mean(X); % vector of means
X = X - X_mean; % deviation scores matrix
S = (X'*X)/M; % sample covariance matrix

% SHRINKAGE TARGET
F = 0; % initialise target covariance matrix
if isempty(shrink) || shrink ~= 0
    F = diag(diag(S)); % shrinkage target
end

% SHRINKAGE COEFFICIENT
if isempty(shrink)  
    % Compute "pi-hat"
    X2 = X.^2; % squared deviation scores
    pi_mat = X2'*X2/M - S.^2; % pi matrix
    pi_diag = diag(pi_mat);
    pi_hat = sum(pi_mat,'all'); % pi matrix
    clear X2 pi_mat
    
    % Compute "rho_hat"
    rho_hat = sum(pi_diag);
    
    % Compute "gamma-hat"
    gamma_hat = norm(S - F,'fro')^2;
  
    % Compute Shrinkage
    kappa_hat = (pi_hat - rho_hat)/gamma_hat;
    shrink = max(0,min(1,kappa_hat/M));
end

% SHRINKED COVARIANCE
C = (1-shrink)*S + shrink*F; % regularised sample covariance matrix
