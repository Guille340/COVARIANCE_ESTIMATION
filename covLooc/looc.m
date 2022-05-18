%  y = looc(alpha,X,S,F,method,alphaScale)
% 
%  DESCRIPTION
%  Returns the mean negative-log likelihood (MNLL) of all the observations 
%  (rows) in the matrix of raw scores X with respect to a covariance 
%  estimate of the form C = S*ALPHA + (1 - ALPHA)*F, where S is the sample 
%  covariance matrix, F is the shrinkage target and ALPHA is the shrinkage 
%  coefficient.
%  
%  The function applies the Leave-One-Out Cross Validation (LOOCV) solution
%  described in Burg et al (1982) and Theiler (2012). The function COVLOOC
%  uses LOOC to arrive at the optimal shrinkage coefficient ALPHA that 
%  minimises the MNLL.
%
%  The algorithm for calculating the MNLL is given by METHOD. There are four
%  solutions available: two exact (full and simplified) and two approximate.
%
%  If an optimisation algorithm is to be used for calculating the optimal
%  shrinkage coefficient, LOG10(ALPHA) can be input instead of ALPHA. In that
%  way, the optimisation algorithm can accurately find the optimal shrinkage
%  coefficient not matter how small without having to dramatically increase 
%  the tolerance factor in the optimisation algorithm (computationally 
%  inefficient). ALPHASCALE gives the user the option to input ALPHA 
%  (ALPHASCALE = 'lin') or LOG10(ALPHA) (ALPHASCALE = 'log').
% 
%  INPUT ARGUMENTS (Fixed)
%  - alpha: shrinkage coefficient
%  - X: matrix of raw scores. One observation per row.
%  - S: sample covariance matrix of X. Avoids having to compute
%    it on every call to LOOC.
%  - F: shrinkage target.
%  - method: string specifying the method used for calculating the MNLL.
%    ¬ 'true': full exact method (computationally inefficient).
%    ¬ 'hl': Hoffbeck & Landgrebe (1996) simplified exact method
%    ¬ 'mc': Monte-Carlo approximate method
%    ¬ 'mm': Mean Mahalanobis approximate method.
%  - 'alphaScale': string specifying the scale of input ALPHA
%    ¬ 'lin': linear (ALPHA)
%    ¬ 'log': logarithmic (10*log10(ALPHA)).
%    
%  OUTPUT VARIABLES
%  - y: mean negative log-likelihood.
%
%  FUNCTION DEPENDENCIES
%  - None
%
%  FUNCTION CALL
%  1. y = looc(alpha,X,S,F,method,alphaScale)
%
%  REFERENCES
%  - Theiler, J. (2012. "The incredible shrinking covariance estimator", 
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

function y = looc(alpha,X,S,F,method,alphaScale)

if strcmp(alphaScale,'log')
    alpha = 10^alpha;
end

[M,N] = size(X); % M = no. observations, N = no. variables

% One-Off Calculations
if ~strcmp(method,'true')
    % beta, G, Gi
    beta = (1 - alpha)/(M - 1);
    G = M*beta*S + alpha*F + 1e-12*eye(size(S)); % added small regularization to avoid singularity
    Gi = G^-1;
    
    % Average Negative-Log Likelihood (1st term)
    term1 = N/2 * log(2*pi); % first term
    
    % Average Negative-Log Likelihood (2nd term, quick, overflows)
%     G_det = det(G); % determinant of G
%     term2 = 1/2 * log(G_det);
    
    % Average Negative-Log Likelihood (2nd term, slow, accurate)
     ev = eig(G); % eigenvalues of G (to avoid overflow) 
     term2 = 1/2 * sum(log(ev)); % equivalent to -1/2 * sum(log(eig(Gi)));
end

% Average Negative-Log Likelihood (3rd term)
switch method        
    case 'true'
        term1 = N/2 * log(2*pi);
        term2 = 0;
        term3 = 0;
        for k = 1:M
            x = X(k,:);
            Sk = M/(M-1)*S - 1/(M-1)*(x'*x);
            R = (1 - alpha)*Sk + alpha*F + 1e-12*eye(size(S));
            
            evr = eig(R);
            term2 = 1/(2*M) * sum(log(evr)) + term2;
            
            Ri = R^-1;
            term3 = 1/(2*M) * x*Ri*x' + term3;
        end
        y = term1 + term2 + term3; % average negative-log likelihood

    case 'hl' % Hoffbeck and Langrebe
        term3 = 0;
        for k = 1:M
            x = X(k,:);
            r = x*Gi*x'; %#ok<*MINV> % same as r = trace(Gi*(x'*x)) or r = sum(Gi.*(x'*x),'all')
            c = 1 - beta*r;
            term3 = log(c) + r/c + term3;
        end
        term3 = term3/(2*M);
        y = term1 + term2 + term3; % average negative-log likelihood
        
    case 'mc' % Monte-Carlo
        M0 = min(M,max(floor(M/10),10)); % reduced no. tests (M >= M0 >= 10)
        term3 = 0;
        for k = 1:M0
            x = X(k,:);
            r = x*Gi*x';
            c = 1 - beta*r;
            term3 = log(c) + r/c + term3;
        end
        term3 = term3/(2*M0);
        y = term1 + term2 + term3; % average negative-log likelihood
         
    case 'mm' % Mean Mahalanobis
        r0 = sum(Gi.*S,'all'); % faster than r0 = trace(Gi*S)
        c0 = 1 - beta*r0;
        term3 =  1/2 * (log(c0) + r0/c0);
        y = term1 + term2 + term3; % average negative-log likelihood
end

y(~isreal(y)) = Inf;

