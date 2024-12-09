clc; clearvars; close all;

%%%%%%%%%%%%%%%%%%% Problem Setup %%%%%%%%%%%%%%%%%%%
Ns        = 7;
rmseNs    = zeros([3,1]);
neesNs    = zeros([3,1]);
timeNs    = zeros([3,1]);
sigG      = 3;
nStates   = 2;

% Initial state and covariance matrices
P0        = eye(nStates);
R         = 1;
Qt        = 1e-2*eye(nStates);
Qf        = Qt;
sqP0      = sqrtm(P0);
sqQt      = sqrtm(Qt);
sqQf      = sqrtm(Qf);
sqR       = sqrt(R);

%%%%%%%%%%%%%%%%%%% Preallocation %%%%%%%%%%%%%%%%%%%
% Truth
nTime     = 50;
nMonte    = 1000;
time      = 0:1:nTime;
nSequence = length(time);
xtruth    = zeros([nStates,nSequence,nMonte]);

% PMF-DWCa
NDWA      = Ns;
xtildeDWA = zeros([nStates,nSequence,nMonte]);
PiiDWA    = zeros([nStates,nSequence,nMonte]);
sneesDWA  = zeros([1,nSequence,nMonte]);
timesDWA  = zeros([1,nSequence,nMonte]);

% PMF-DWCn
NDWN      = Ns;
xtildeDWN = zeros([nStates,nSequence,nMonte]);
PiiDWN    = zeros([nStates,nSequence,nMonte]);
sneesDWN  = zeros([1,nSequence,nMonte]);
timesDWN  = zeros([1,nSequence,nMonte]);

% SMF
NSMF      = Ns;
xtildeSMF = zeros([nStates,nSequence,nMonte]);
PiiSMF    = zeros([nStates,nSequence,nMonte]);
sneesSMF  = zeros([1,nSequence,nMonte]);
timesSMF  = zeros([1,nSequence,nMonte]);

%%%%%%%%%%%%%%%%%%% Monte Carlo Loop %%%%%%%%%%%%%%%%%%%
parfor iMonte = 1:nMonte; rng(iMonte,'twister');

    %%%%%%%%%%%%%%%%%%% Initialization %%%%%%%%%%%%%%%%%%%
    % Truth
    xkm1        = 0;
    xk          = [0;0];
    xhatk       = xk + sqP0*randn([nStates,1]);

    % PMF-DWCa
    Xhatkm1DWA  = 0;
    xhatkDWA    = xhatk;
    PhatkDWA    = P0;
    XhatkDWA    = utils.gridP(xhatk,P0,NDWA,sigG);
    ngridDWA    = size(XhatkDWA,2);
    whatkm1DWA  = 0;
    whatkDWA    = zeros([1,ngridDWA]);
    for iPoint = 1:ngridDWA
        nuxk             = XhatkDWA(:,iPoint)-xhatk;
        whatkDWA(iPoint) = (-0.5*nuxk.'*(P0\nuxk));
    end
    m           = max(whatkDWA);
    whatkDWA    = exp(whatkDWA - (m + log(sum(exp(whatkDWA - m)))));
    whatkDWA    = whatkDWA/sum(whatkDWA);

    % PMF-DWCn
    Xhatkm1DWN  = 0;
    xhatkDWN    = xhatk;
    PhatkDWN    = P0;
    XhatkDWN    = utils.gridP(xhatk,P0,NDWN,sigG);
    ngridDWN    = size(XhatkDWN,2);
    whatkm1DWN  = 0;
    whatkDWN    = zeros([1,ngridDWN]);
    for iPoint = 1:ngridDWN
        nuxk             = XhatkDWN(:,iPoint)-xhatk;
        whatkDWN(iPoint) = (-0.5*nuxk.'*(P0\nuxk));
    end
    m           = max(whatkDWN);
    whatkDWN    = exp(whatkDWN - (m + log(sum(exp(whatkDWN - m)))));
    whatkDWN    = whatkDWN/sum(whatkDWN);

    % SMF
    Xhatkm1SMF  = 0;
    xhatkSMF    = xhatk;
    PhatkSMF    = P0;
    XhatkSMF    = utils.gridP(xhatk,P0,NSMF,sigG);
    ngridSMF    = size(XhatkSMF,2);
    whatkm1SMF  = 0;
    whatkSMF    = zeros([1,ngridSMF]);
    whatkSMFi   = zeros([1,ngridSMF]);
    for iPoint = 1:ngridSMF
        nuxk             = XhatkSMF(:,iPoint)-xhatk;
        whatkSMF(iPoint) = (-0.5*nuxk.'*(P0\nuxk));
    end
    m           = max(whatkSMF);
    whatkSMF    = exp(whatkSMF - (m + log(sum(exp(whatkSMF - m)))));
    whatkSMF    = whatkSMF/sum(whatkSMF);
    betaSMF     = 0.4;

    %%%%%%%%%%%%%%%%%%% Time Loop %%%%%%%%%%%%%%%%%%%
    for iSequence = 1:nSequence

        if iSequence == 1

            %%%%%%%%%%%%%%%%%%% Saving metrics %%%%%%%%%%%%%%%%%%%
            % Truth
            xtruth(:,iSequence,iMonte)    = xk;

            % PMF-DWCa
            xtildekDWA                    = xk - xhatkDWA;
            xtildeDWA(:,iSequence,iMonte) = xtildekDWA;
            PiiDWA(:,iSequence,iMonte)    = diag(PhatkDWA);
            sneesDWA(:,iSequence,iMonte)  = xtildekDWA.'*pinv(PhatkDWA)*xtildekDWA;

            % PMF-DWCn
            xtildekDWN                    = xk - xhatkDWN;
            xtildeDWN(:,iSequence,iMonte) = xtildekDWN;
            PiiDWN(:,iSequence,iMonte)    = diag(PhatkDWN);
            sneesDWN(:,iSequence,iMonte)  = xtildekDWN.'*pinv(PhatkDWN)*xtildekDWN;

            % SMF
            xtildekSMF                    = xk - xhatkSMF;
            xtildeSMF(:,iSequence,iMonte) = xtildekSMF;
            PiiSMF(:,iSequence,iMonte)    = diag(PhatkSMF);
            sneesSMF(:,iSequence,iMonte)  = xtildekSMF.'*pinv(PhatkSMF)*xtildekSMF;
        
            %%%%%%%%%%%%%%%%%%% Update time index %%%%%%%%%%%%%%%%%%%
            % Truth
            xkm1  = xk;

            % PMF-DWCa
            Xhatkm1DWA = XhatkDWA;
            whatkm1DWA = whatkDWA;

            % PMF-DWCn
            Xhatkm1DWN = XhatkDWN;
            whatkm1DWN = whatkDWN;

            % SMF
            Xhatkm1SMF = XhatkSMF;
            whatkm1SMF = whatkSMF;   

            continue
        end

        %%%%%%%%%%%%%%%%%% Truth %%%%%%%%%%%%%%%%%%%
        xk = utils.f(xkm1) + sqQt*randn([nStates,1]);
        yk = utils.h(xk) + sqR*randn();

        %%%%%%%%%%%%%%%%%%% PMF-DWCa %%%%%%%%%%%%%%%%%%%
        tic;
        [XhatkDWA,whatkDWA,xhatkDWA,PhatkDWA] = utils.DWF(Xhatkm1DWA,whatkm1DWA,Qf,NDWA,sigG,yk,R,true);
        timesDWA(:,iSequence,iMonte) = toc;

        %%%%%%%%%%%%%%%%%%% PMF-DWCn %%%%%%%%%%%%%%%%%%%
        tic;
        [XhatkDWN,whatkDWN,xhatkDWN,PhatkDWN] = utils.DWF(Xhatkm1DWN,whatkm1DWN,Qf,NDWN,sigG,yk,R,false);
        timesDWN(:,iSequence,iMonte) = toc;

        %%%%%%%%%%%%%%%%%%% SMF %%%%%%%%%%%%%%%%%%%
        tic;
        [XhatkSMF,whatkSMF,xhatkSMF,PhatkSMF] = utils.SMS(Xhatkm1SMF,whatkm1SMF,Qf,NSMF,sigG,yk,R,betaSMF);
        timesSMF(:,iSequence,iMonte) = toc;

        %%%%%%%%%%%%%%%%%%% Saving metrics %%%%%%%%%%%%%%%%%%%
        % Truth
        xtruth(:,iSequence,iMonte)    = xk;

        % PMF-DWCa
        xtildekDWA                    = xk - xhatkDWA;
        xtildeDWA(:,iSequence,iMonte) = xtildekDWA;
        PiiDWA(:,iSequence,iMonte)    = diag(PhatkDWA);
        sneeskDWA                     = xtildekDWA.'*pinv(PhatkDWA)*xtildekDWA;
        if sneeskDWA > 1e4;sneeskDWA  = nan; end
        sneesDWA(:,iSequence,iMonte)  = sneeskDWA;

        % PMF-DWCn
        xtildekDWN                    = xk - xhatkDWN;
        xtildeDWN(:,iSequence,iMonte) = xtildekDWN;
        PiiDWN(:,iSequence,iMonte)    = diag(PhatkDWN);
        sneeskDWN                     = xtildekDWN.'*pinv(PhatkDWN)*xtildekDWN;
        if sneeskDWN > 1e4;sneeskDWN  = nan; end
        sneesDWN(:,iSequence,iMonte)  = sneeskDWN;

        % SMF
        xtildekSMF                    = xk - xhatkSMF;
        xtildeSMF(:,iSequence,iMonte) = xtildekSMF;
        PiiSMF(:,iSequence,iMonte)    = diag(PhatkSMF);
        sneeskSMF                     = xtildekSMF.'*pinv(PhatkSMF)*xtildekSMF;
        if sneeskSMF > 1e4;sneeskSMF  = nan; end
        sneesSMF(:,iSequence,iMonte)  = sneeskSMF;

        %%%%%%%%%%%%%%%%%%% Update time index %%%%%%%%%%%%%%%%%%%
        % Truth
        xkm1  = xk;

        % PMF-DWCa
        Xhatkm1DWA = XhatkDWA;
        whatkm1DWA = whatkDWA;

        % PMF-DWCn
        Xhatkm1DWN = XhatkDWN;
        whatkm1DWN = whatkDWN;

        % SMF
        Xhatkm1SMF = XhatkSMF;
        whatkm1SMF = whatkSMF;
    end
end

% RMSE
rmseNs(1) = mean(mean(sqrt((mean(xtildeDWA.^2,1))),3));
rmseNs(2) = mean(mean(sqrt((mean(xtildeDWN.^2,1))),3));
rmseNs(3) = mean(mean(sqrt((mean(xtildeSMF.^2,1))),3));

% SNEES
neesNs(1) = mean(sneesDWA,'all','omitnan')./nStates;
neesNs(2) = mean(sneesDWN,'all','omitnan')./nStates;
neesNs(3) = mean(sneesSMF,'all','omitnan')./nStates;

% Time
timeNs(1) = mean(timesDWA,'all');
timeNs(2) = mean(timesDWN,'all');
timeNs(3) = mean(timesSMF,'all');

% Display results
tableRes = array2table([rmseNs neesNs timeNs],'RowNames',{'DWA','DWN','SMF'},'VariableNames',{'RMSE','SNEES','Time'});
format shortE
clc; disp(tableRes);