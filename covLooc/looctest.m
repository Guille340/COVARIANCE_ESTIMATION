
% filtObj = designfilt('bandpassfir','FilterOrder',20,'CutoffFrequency1',0.45,'CutoffFrequency2',0.55);
N = 500;
M = 25:50:750; % vector of number of observations
bandType = 'TOB';
sampleRateForDetection = 2;
filtMode = 'filtfilt';

switch bandType
    case 'BDB'
        cutoffFreqs = sampleRateForDetection/4 * [0 2];
    case 'OOB'
        cutoffFreqs = sampleRateForDetection/4 * [2^(-1/2) 2^(1/2)];
    case 'TOB'
        cutoffFreqs = sampleRateForDetection/4 * [2^(-1/6) 2^(1/6)];
end
DigitalFilter = digitalSingleFilterDesign(sampleRateForDetection,cutoffFreqs);
tic
F = targetCovariance(N,[],DigitalFilter,filtMode); % target covariance based on filtered WGN
toc
X = randn(max(M),N);
for m = 1:max(M)
    X(m,:) = digitalSingleFilter(DigitalFilter,X(m,:),'MetricsOutput',...
            false,'FilterMode',filtMode,'DataWrap',true);
end

for k = 1:length(M)
    M0 = M(k);
    X0 = X(1:M0,:);
%     X0 = X0./sqrt(mean(sum(X0.^2)/M0));
%     S0 = X0'*X0/M0;
%     F0 = F*mean(diag(S0))/mean(diag(F));
%     F0 = F*sqrt(mean(sum(X0.^2)/M0));
           
    [~,shrink0(k)] = covLooc(X0,[],F);
    [~,shrink1(k)] = covLooc(X0,[],'origdiag');
    [~,shrink2(k)] = covDiag(X0);
end

alphaLog = -10:0.05:0;
for m = 1:length(alphaLog)
    fprintf('%d/%d\n',m,length(alphaLog))
    y(m) = looc(alphaLog(m),X0,S0,F0,'hl','log');
end


% alphaLog = -10:0.2:0;
% for m = 1:length(alphaLog)
%     fprintf('%d/%d\n',m,length(alphaLog))
%     for n = 1:length(alphaLog)
%         y(m,n) = looc([alphaLog(m),alphaLog(n)],X,S,F,'hl','log');
%     end
% end


% 
% 
% 
% figure
% subplot(1,2,1)
% hold on
% plot(alpha,term1 + term2 + term3,'k','linewidth',1.5)
% plot(alpha,term1,'b')
% plot(alpha,term2,'r')
% plot(alpha,term3,'g')
% xlabel('\alpha')
% ylabel('Neg-Log Likelihood')
% set(gca,'XScale','log')
% box on
% title(sprintf('N = %d, M = %d',M,N))
% legend({'y = term1 + term2 + term3','term1 = N/2*log(2*pi)',...
%     'term2 = 1/2*log(det(G))','term3 = log(c) + r/c'},...
%     'location','best')
% 
% subplot(1,2,2)
% hold on
% plot(alpha,term1 + term2 + term3,'k','linewidth',1.5)
% xlabel('\alpha')
% ylabel('Neg-Log Likelihood')
% set(gca,'XScale','log')
% axis([1e-10 1 120 160])
% box on
% title(sprintf('N = %d, M = %d \\rm(zoom)',M,N))
% legend({'y = term1 + term2 + term3'},'location','southwest')
% set(gcf,'units','normalized','outerposition',[0.05 0.1 0.9 0.8])
% 
% figName = sprintf('N = %d, M = %d',M,N);
% set(gcf,'positionmode','auto')
% print(figName,'-dpng','-r250')
% savefig(figName)