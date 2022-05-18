%  [C,shrink] = covParam2(X,varargin)
% 
%  DESCRIPTION
%  Computes the covariance matrix from a matrix of raw scores X using two-
%  parameter shrinkage method from Ledoit.
%
%  The shrinkage method is suitable for high-dimensional problems (large number
%  of variables N) with a small number of samples (few observations M). The 
%  shrinkage approach attempts to address the low accuracy of the sample 
%  covariance matrix (X'*X/M) when M < N by introducing bias. 
%
%  COVPARAM2 returns the shrunk covariance matrix C and shrinkage coefficient 
%  SHRINK. The shrunk covariance C is estimated as a linear combination of
%  the sample covariance matrix S and a shrinkage target matrix F by means of 
%  the shrinkage coefficient SHRINK. C computed as follows:
% 
%       C = (1-RHO)*S + RHO*F
%
%  In this case, in the target matrix F the diagonal and off-diagonal entries 
%  are respectively equal to the mean variance and mean covariance of the 
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
%  - 'OffDiag': TRUE to include the contribution from the off-diagonal entries.
%    The effect of this contribution is relatively small but the calculation
%    can be computationally intensive. By default, OFFDIAG = FALSE.
%     
%    
%  OUTPUT VARIABLES
%  - C: covariance estimate.
%  - shrink: shrinkage coefficient.
%
%  FUNCTION DEPENDENCIES
%  - None
%
%  FUNCTION CALL
%  1. [C,shrink] = covParam2(X)
%  2. [C,shrink] = covParam2(...,PROPERTY,VALUE)
%       ¬ 'ShrinkCoef'
%
%  REFERENCES
%  - http://ledoit.net/cov2para.m
%  - "Estimation of Covariance Matrices", Wikipedia article.
%    https://en.wikipedia.org/wiki/Estimation_of_covariance_matrices

%  VERSION 1.0
%  Date: 28 Oct 2021
%  Author: Guillermo Jimenez Arranz
%  Email: gjarranz@gmail.com

function [C,shrink] = covParam2(X,varargin)

% INPUT ARGUMENTS
% Verify number of Input Arguments
nFixArg = 1;
nProArg = 4;
narginchk(nFixArg,nFixArg + nProArg)

% Verify Number of Property/Value Pairs
if rem(nargin - nFixArg,2)
    error('Variable input arguments must come in pairs (PROPERTY,VALUE)')
end

% Extract and Verify Input Properties
validProperties = {'shrinkcoef','offdiag'};
properties = lower(varargin(1:2:end));
if any(~ismember(properties,validProperties))
    error('One or more input properties are not recognised')
end

% Default Input Values
shrink = [];
offdiag = false;

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
         case 'offdiag'
            offdiag = values{m};    
            if ~islogical(offdiag) && ~any(offdiag == [0 1])
                offdiag = 0;
                warning(['Non-supported value for PROPERTY = '...
                    '''OffDiag''. A value of 0 will be used'])
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
    meanVar = trace(S)/N;
    meanCov = sum(S(~eye(N)),'all')/(N*(N-1));
    F = meanVar*eye(N) + meanCov*(~eye(N)); % shrinkage target
end

if isempty(shrink)
    % Compute "c"
    c = norm(S - F,'fro')^2;
    
    % Compute "p"
    X2 = X.^2; % squared deviation scores
    X2_mean = mean(X2); % vector of means
    X2 = X2 - X2_mean; % deviation scores matrix
    p = sum(X2'*X2/M - S.^2,'all'); % pi matrix
    
    % Compute "r" (diagonal)
    r_diag = sum(X2'*X2/M,'all')/N;

    % Compute "r" (off-diagonal)
    r_off = 0;
    if offdiag
      nVar = (N-1)*N/2; % number of variables
      Z = ones(M,nVar);
      col = 1;
      for i = 1:(N-1)
        for j = (i+1):N
          Z(:,col) = X(:,i).*X(:,j);
          col = col + 1;
        end  
      end
      r_off = 4/(N*(N-1)) * sum(Z'*Z/M,'all')/M;
    end
    r = r_diag + r_off;
    
    % Compute Shrinkage
      k = (p-r)/c;
      shrink = max(0,min(1,k/M));
end

% SHRINKED COVARIANCE
C = (1-shrink)*S + shrink*F;
