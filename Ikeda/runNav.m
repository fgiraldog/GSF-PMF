clc; clearvars; close all;
%%%%%%%%%%%%%%%%%%% Colors Setup %%%%%%%%%%%%%%%%%%%
C = [...
    0.2667    0.4667    0.6667; % Blue
    0.9333    0.4000    0.4667; % Red
    0.6667    0.2000    0.4667; % Purple
    0.1333    0.5333    0.2000; % Green
    0.8000    0.7333    0.2667; % Yellow
    0.6350    0.0780    0.1840; % Dark Red
    0.4000    0.8000    0.9333; % Cyan
    ];

%%%%%%%%%%%%%%%%%%% Grid Study Setup %%%%%%%%%%%%%%%%%%%
Ns     = 7:2:19;
rmseNs = zeros([7,length(Ns)]);
neesNs = zeros([7,length(Ns)]);
timeNs = zeros([7,length(Ns)]);
sigG   = 3;

for iN = length(Ns):-1:1
    %%%%%%%%%%%%%%%%%%% Problem Setup %%%%%%%%%%%%%%%%%%%
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

    % PMF
    NPMF      = Ns(iN);
    xtildePMF = zeros([nStates,nSequence,nMonte]);
    PiiPMF    = zeros([nStates,nSequence,nMonte]);
    sneesPMF  = zeros([1,nSequence,nMonte]);
    timesPMF  = zeros([1,nSequence,nMonte]);

    % PMF-BER
    NBMF      = Ns(iN);
    xtildeBMF = zeros([nStates,nSequence,nMonte]);
    PiiBMF    = zeros([nStates,nSequence,nMonte]);
    sneesBMF  = zeros([1,nSequence,nMonte]);
    timesBMF  = zeros([1,nSequence,nMonte]);

    % PMF-UKF
    NUMF      = Ns(iN);
    xtildeUMF = zeros([nStates,nSequence,nMonte]);
    PiiUMF    = zeros([nStates,nSequence,nMonte]);
    sneesUMF  = zeros([1,nSequence,nMonte]);
    timesUMF  = zeros([1,nSequence,nMonte]);

    % PMF-DHF
    NDMF      = Ns(iN);
    xtildeDMF = zeros([nStates,nSequence,nMonte]);
    PiiDMF    = zeros([nStates,nSequence,nMonte]);
    sneesDMF  = zeros([1,nSequence,nMonte]);
    timesDMF  = zeros([1,nSequence,nMonte]);

    % FMF1
    NRMF      = Ns(iN);
    xtildeRMF = zeros([nStates,nSequence,nMonte]);
    PiiRMF    = zeros([nStates,nSequence,nMonte]);
    sneesRMF  = zeros([1,nSequence,nMonte]);
    timesRMF  = zeros([1,nSequence,nMonte]);

    % FMF2
    NFMF      = Ns(iN);
    xtildeFMF = zeros([nStates,nSequence,nMonte]);
    PiiFMF    = zeros([nStates,nSequence,nMonte]);
    sneesFMF  = zeros([1,nSequence,nMonte]);
    timesFMF  = zeros([1,nSequence,nMonte]);
   
    % SMF
    NSMF      = Ns(iN);
    xtildeSMF = zeros([nStates,nSequence,nMonte]);
    PiiSMF    = zeros([nStates,nSequence,nMonte]);
    sneesSMF  = zeros([1,nSequence,nMonte]);
    timesSMF  = zeros([1,nSequence,nMonte]);

    %%%%%%%%%%%%%%%%%%% Monte Carlo Loop %%%%%%%%%%%%%%%%%%%
    for iMonte = 1:nMonte; rng(iMonte,'twister');

        %%%%%%%%%%%%%%%%%%% Initialization %%%%%%%%%%%%%%%%%%%
        % Truth
        xkm1        = 0;
        xk          = [0;0];
        xhatk       = xk + sqP0*randn([nStates,1]);

        % PMF
        Xhatkm1PMF  = 0;
        xhatkPMF    = xhatk;
        PhatkPMF    = P0;
        XhatkPMF    = utils.gridP(xhatk,P0,NPMF,sigG);
        ngridPMF    = size(XhatkPMF,2);
        whatkm1PMF  = 0;
        whatkPMF    = zeros([1,ngridPMF]);
        for iPoint = 1:ngridPMF
            nuxk             = XhatkPMF(:,iPoint)-xhatk;
            whatkPMF(iPoint) = (-0.5*nuxk.'*(P0\nuxk));
        end
        m           = max(whatkPMF);
        whatkPMF    = exp(whatkPMF - (m + log(sum(exp(whatkPMF - m)))));
        whatkPMF    = whatkPMF/sum(whatkPMF);

        % PMF-BER
        Xhatkm1BMF  = 0;
        xhatkBMF    = xhatk;
        PhatkBMF    = P0;
        XhatkBMF    = utils.gridP(xhatk,P0,NBMF,sigG);
        ngridBMF    = size(XhatkBMF,2);
        whatkm1BMF  = 0;
        whatkBMF    = zeros([1,ngridBMF]);
        for iPoint = 1:ngridBMF
            nuxk             = XhatkBMF(:,iPoint)-xhatk;
            whatkBMF(iPoint) = (-0.5*nuxk.'*(P0\nuxk));
        end
        m           = max(whatkBMF);
        whatkBMF    = exp(whatkBMF - (m + log(sum(exp(whatkBMF - m)))));
        whatkBMF    = whatkBMF/sum(whatkBMF);

        % PMF-UKF
        Xhatkm1UMF  = 0;
        xhatkUMF    = xhatk;
        PhatkUMF    = P0;
        XhatkUMF    = utils.gridP(xhatk,P0,NUMF,sigG);
        ngridUMF    = size(XhatkUMF,2);
        whatkm1UMF  = 0;
        whatkUMF    = zeros([1,ngridUMF]);
        for iPoint = 1:ngridUMF
            nuxk             = XhatkUMF(:,iPoint)-xhatk;
            whatkUMF(iPoint) = (-0.5*nuxk.'*(P0\nuxk));
        end
        m           = max(whatkUMF);
        whatkUMF    = exp(whatkUMF - (m + log(sum(exp(whatkUMF - m)))));
        whatkUMF    = whatkUMF/sum(whatkUMF);

        % PMF-DHF
        Xhatkm1DMF  = 0;
        xhatkDMF    = xhatk;
        PhatkDMF    = P0;
        XhatkDMF    = utils.gridP(xhatk,P0,NDMF,sigG);
        ngridDMF    = size(XhatkDMF,2);
        whatkm1DMF  = 0;
        whatkDMF    = zeros([1,ngridDMF]);
        for iPoint = 1:ngridDMF
            nuxk             = XhatkDMF(:,iPoint)-xhatk;
            whatkDMF(iPoint) = (-0.5*nuxk.'*(P0\nuxk));
        end
        m           = max(whatkDMF);
        whatkDMF    = exp(whatkDMF - (m + log(sum(exp(whatkDMF - m)))));
        whatkDMF    = whatkDMF/sum(whatkDMF);
     
        % FMF1
        Xhatkm1RMF  = 0;
        xhatkRMF    = xhatk;
        PhatkRMF    = P0;
        XhatkRMF    = utils.gridP(xhatk,P0,NRMF,sigG);
        ngridRMF    = size(XhatkRMF,2);
        whatkm1RMF  = 0;
        whatkRMF    = zeros([1,ngridRMF]);
        for iPoint = 1:ngridRMF
            nuxk             = XhatkRMF(:,iPoint)-xhatk;
            whatkRMF(iPoint) = (-0.5*nuxk.'*(P0\nuxk));
        end
        m           = max(whatkRMF);
        whatkRMF    = exp(whatkRMF - (m + log(sum(exp(whatkRMF - m)))));
        whatkRMF    = whatkRMF/sum(whatkRMF);

        % FMF2
        Xhatkm1FMF  = 0;
        xhatkFMF    = xhatk;
        PhatkFMF    = P0;
        XhatkFMF    = utils.gridP(xhatk,P0,NFMF,sigG);
        ngridFMF    = size(XhatkFMF,2);
        whatkm1FMF  = 0;
        whatkFMF    = zeros([1,ngridFMF]);
        for iPoint = 1:ngridFMF
            nuxk             = XhatkFMF(:,iPoint)-xhatk;
            whatkFMF(iPoint) = (-0.5*nuxk.'*(P0\nuxk));
        end
        m           = max(whatkFMF);
        whatkFMF    = exp(whatkFMF - (m + log(sum(exp(whatkFMF - m)))));
        whatkFMF    = whatkFMF/sum(whatkFMF);

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
        betaSMF     = 0.2;

        %%%%%%%%%%%%%%%%%%% Time Loop %%%%%%%%%%%%%%%%%%%
        for iSequence = 1:nSequence

            if iSequence == 1

                %%%%%%%%%%%%%%%%%%% Saving metrics %%%%%%%%%%%%%%%%%%%
                % Truth
                xtruth(:,iSequence,iMonte)    = xk;

                % PMF
                xtildekPMF                    = xk - xhatkPMF;
                xtildePMF(:,iSequence,iMonte) = xtildekPMF;
                PiiPMF(:,iSequence,iMonte)    = diag(PhatkPMF);
                sneesPMF(:,iSequence,iMonte)  = xtildekPMF.'*pinv(PhatkPMF)*xtildekPMF;

                % PMF-BER
                xtildekBMF                    = xk - xhatkBMF;
                xtildeBMF(:,iSequence,iMonte) = xtildekBMF;
                PiiBMF(:,iSequence,iMonte)    = diag(PhatkBMF);
                sneesBMF(:,iSequence,iMonte)  = xtildekBMF.'*pinv(PhatkBMF)*xtildekBMF;

                % PMF-UKF
                xtildekUMF                    = xk - xhatkUMF;
                xtildeUMF(:,iSequence,iMonte) = xtildekUMF;
                PiiUMF(:,iSequence,iMonte)    = diag(PhatkUMF);
                sneesUMF(:,iSequence,iMonte)  = xtildekUMF.'*pinv(PhatkUMF)*xtildekUMF;

                % PMF-DHF
                xtildekDMF                    = xk - xhatkDMF;
                xtildeDMF(:,iSequence,iMonte) = xtildekDMF;
                PiiDMF(:,iSequence,iMonte)    = diag(PhatkDMF);
                sneesDMF(:,iSequence,iMonte)  = xtildekDMF.'*pinv(PhatkDMF)*xtildekDMF;
                
                % FMF1
                xtildekRMF                    = xk - xhatkRMF;
                xtildeRMF(:,iSequence,iMonte) = xtildekRMF;
                PiiRMF(:,iSequence,iMonte)    = diag(PhatkRMF);
                sneesRMF(:,iSequence,iMonte)  = xtildekRMF.'*pinv(PhatkRMF)*xtildekRMF;

                % FMF2
                xtildekFMF                    = xk - xhatkFMF;
                xtildeFMF(:,iSequence,iMonte) = xtildekFMF;
                PiiFMF(:,iSequence,iMonte)    = diag(PhatkFMF);
                sneesFMF(:,iSequence,iMonte)  = xtildekFMF.'*pinv(PhatkFMF)*xtildekFMF;

                % SMF
                xtildekSMF                    = xk - xhatkSMF;
                xtildeSMF(:,iSequence,iMonte) = xtildekSMF;
                PiiSMF(:,iSequence,iMonte)    = diag(PhatkSMF);
                sneesSMF(:,iSequence,iMonte)  = xtildekSMF.'*pinv(PhatkSMF)*xtildekSMF;
            
                %%%%%%%%%%%%%%%%%%% Update time index %%%%%%%%%%%%%%%%%%%
                % Truth
                xkm1  = xk;

                % PMF
                Xhatkm1PMF = XhatkPMF;
                whatkm1PMF = whatkPMF;

                % PMF-BER
                Xhatkm1BMF = XhatkBMF;
                whatkm1BMF = whatkBMF;

                % PMF-UKF
                Xhatkm1UMF = XhatkUMF;
                whatkm1UMF = whatkUMF;

                % PMF-DHF
                Xhatkm1DMF = XhatkDMF;
                whatkm1DMF = whatkDMF;

                % FMF1
                Xhatkm1RMF = XhatkRMF;
                whatkm1RMF = whatkRMF;

                % FMF2
                Xhatkm1FMF = XhatkFMF;
                whatkm1FMF = whatkFMF;

                % SMF
                Xhatkm1SMF = XhatkSMF;
                whatkm1SMF = whatkSMF;   

                continue
            end

            %%%%%%%%%%%%%%%%%% Truth %%%%%%%%%%%%%%%%%%%
            xk = utils.f(xkm1) + sqQt*randn([nStates,1]);
            yk = utils.h(xk) + sqR*randn();

            %%%%%%%%%%%%%%%%%%% PMF %%%%%%%%%%%%%%%%%%%
            tic;
            [XhatkPMF,whatkPMF,xhatkPMF,PhatkPMF] = utils.PMF(Xhatkm1PMF,whatkm1PMF,Qf,NPMF,sigG,yk,R);
            timesPMF(:,iSequence,iMonte) = toc;

            %%%%%%%%%%%%%%%%%%% PMF-BER %%%%%%%%%%%%%%%%%%%
            tic;
            [XhatkBMF,whatkBMF,xhatkBMF,PhatkBMF] = utils.BMF(Xhatkm1BMF,whatkm1BMF,Qf,NBMF,sigG,yk,R);
            timesBMF(:,iSequence,iMonte) = toc;

            %%%%%%%%%%%%%%%%%%% PMF-UKF %%%%%%%%%%%%%%%%%%%
            tic;
            [XhatkUMF,whatkUMF,xhatkUMF,PhatkUMF] = utils.UMF(Xhatkm1UMF,whatkm1UMF,Qf,NUMF,sigG,yk,R);
            timesUMF(:,iSequence,iMonte) = toc;

            %%%%%%%%%%%%%%%%%%% PMF-DHF %%%%%%%%%%%%%%%%%%%
            tic;
            [XhatkDMF,whatkDMF,xhatkDMF,PhatkDMF] = utils.DMF(Xhatkm1DMF,whatkm1DMF,Qf,NDMF,sigG,yk,R);
            timesDMF(:,iSequence,iMonte) = toc;

            %%%%%%%%%%%%%%%%%%% FMF1 %%%%%%%%%%%%%%%%%%%
            tic;
            [XhatkRMF,whatkRMF,xhatkRMF,PhatkRMF] = utils.RMF(Xhatkm1RMF,whatkm1RMF,Qf,NRMF,sigG,yk,R);
            timesRMF(:,iSequence,iMonte) = toc;

            %%%%%%%%%%%%%%%%%%% FMF2 %%%%%%%%%%%%%%%%%%%
            tic;
            [XhatkFMF,whatkFMF,xhatkFMF,PhatkFMF] = utils.FMF(Xhatkm1FMF,whatkm1FMF,Qf,NFMF,sigG,yk,R);
            timesFMF(:,iSequence,iMonte) = toc;

            %%%%%%%%%%%%%%%%%%% SMF %%%%%%%%%%%%%%%%%%%
            tic;
            [XhatkSMF,whatkSMF,xhatkSMF,PhatkSMF] = utils.SMF(Xhatkm1SMF,whatkm1SMF,Qf,NSMF,sigG,yk,R,betaSMF);
            timesSMF(:,iSequence,iMonte) = toc;

            %%%%%%%%%%%%%%%%%%% Saving metrics %%%%%%%%%%%%%%%%%%%
            % Truth
            xtruth(:,iSequence,iMonte)    = xk;

            % PMF
            xtildekPMF                    = xk - xhatkPMF;
            xtildePMF(:,iSequence,iMonte) = xtildekPMF;
            PiiPMF(:,iSequence,iMonte)    = diag(PhatkPMF);
            sneeskPMF                     = xtildekPMF.'*pinv(PhatkPMF)*xtildekPMF;
            if sneeskPMF > 1e4;sneeskPMF  = nan; end
            sneesPMF(:,iSequence,iMonte)  = sneeskPMF;

            % PMF-BER
            xtildekBMF                    = xk - xhatkBMF;
            xtildeBMF(:,iSequence,iMonte) = xtildekBMF;
            PiiBMF(:,iSequence,iMonte)    = diag(PhatkBMF);
            sneeskBMF                     = xtildekBMF.'*pinv(PhatkBMF)*xtildekBMF;
            if sneeskBMF > 1e4;sneeskBMF  = nan; end
            sneesBMF(:,iSequence,iMonte)  = sneeskBMF;
                      
            % PMF-UKF
            xtildekUMF                    = xk - xhatkUMF;
            xtildeUMF(:,iSequence,iMonte) = xtildekUMF;
            PiiUMF(:,iSequence,iMonte)    = diag(PhatkUMF);
            sneeskUMF                     = xtildekUMF.'*pinv(PhatkUMF)*xtildekUMF;
            if sneeskUMF > 1e4;sneeskUMF  = nan; end
            sneesUMF(:,iSequence,iMonte)  = sneeskUMF;

            % PMF-DHF
            xtildekDMF                    = xk - xhatkDMF;
            xtildeDMF(:,iSequence,iMonte) = xtildekDMF;
            PiiDMF(:,iSequence,iMonte)    = diag(PhatkDMF);
            sneeskDMF                     = xtildekDMF.'*pinv(PhatkDMF)*xtildekDMF;
            if sneeskDMF > 1e4;sneeskDMF  = nan; end
            sneesDMF(:,iSequence,iMonte)  = sneeskDMF;

            % FMF1
            xtildekRMF                    = xk - xhatkRMF;
            xtildeRMF(:,iSequence,iMonte) = xtildekRMF;
            PiiRMF(:,iSequence,iMonte)    = diag(PhatkRMF);
            sneeskRMF                     = xtildekRMF.'*pinv(PhatkRMF)*xtildekRMF;
            if sneeskRMF > 1e4;sneeskRMF  = nan; end
            sneesRMF(:,iSequence,iMonte)  = sneeskRMF;

            % FMF2
            xtildekFMF                    = xk - xhatkFMF;
            xtildeFMF(:,iSequence,iMonte) = xtildekFMF;
            PiiFMF(:,iSequence,iMonte)    = diag(PhatkFMF);
            sneeskFMF                     = xtildekFMF.'*pinv(PhatkFMF)*xtildekFMF;
            if sneeskFMF > 1e4;sneeskFMF  = nan; end
            sneesFMF(:,iSequence,iMonte)  = sneeskFMF;

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

            % PMF
            Xhatkm1PMF = XhatkPMF;
            whatkm1PMF = whatkPMF;

            % PMF-BER
            Xhatkm1BMF = XhatkBMF;
            whatkm1BMF = whatkBMF;

            % PMF-UKF
            Xhatkm1UMF = XhatkUMF;
            whatkm1UMF = whatkUMF;

             % PMF-DHF
            Xhatkm1DMF = XhatkDMF;
            whatkm1DMF = whatkDMF;

            % FMF1
            Xhatkm1RMF = XhatkRMF;
            whatkm1RMF = whatkRMF;

            % FMF2
            Xhatkm1FMF = XhatkFMF;
            whatkm1FMF = whatkFMF;

            % SMF
            Xhatkm1SMF = XhatkSMF;
            whatkm1SMF = whatkSMF;
        end
    end

    % RMSE
    rmseNs(1,iN) = mean(mean(sqrt((mean(xtildePMF.^2,1))),3));
    rmseNs(2,iN) = mean(mean(sqrt((mean(xtildeBMF.^2,1))),3));
    rmseNs(3,iN) = mean(mean(sqrt((mean(xtildeUMF.^2,1))),3));
    rmseNs(4,iN) = mean(mean(sqrt((mean(xtildeDMF.^2,1))),3));
    rmseNs(5,iN) = mean(mean(sqrt((mean(xtildeRMF.^2,1))),3));
    rmseNs(6,iN) = mean(mean(sqrt((mean(xtildeFMF.^2,1))),3));
    rmseNs(7,iN) = mean(mean(sqrt((mean(xtildeSMF.^2,1))),3));

    % SNEES
    neesNs(1,iN) = mean(sneesPMF,'all','omitnan')./nStates;
    neesNs(2,iN) = mean(sneesBMF,'all','omitnan')./nStates;
    neesNs(3,iN) = mean(sneesUMF,'all','omitnan')./nStates;
    neesNs(4,iN) = mean(sneesDMF,'all','omitnan')./nStates;
    neesNs(5,iN) = mean(sneesRMF,'all','omitnan')./nStates;
    neesNs(6,iN) = mean(sneesFMF,'all','omitnan')./nStates;
    neesNs(7,iN) = mean(sneesSMF,'all','omitnan')./nStates;

    % Time
    timeNs(1,iN) = mean(timesPMF,'all');
    timeNs(2,iN) = mean(timesBMF,'all');
    timeNs(3,iN) = mean(timesUMF,'all');
    timeNs(4,iN) = mean(timesDMF,'all');
    timeNs(5,iN) = mean(timesRMF,'all');
    timeNs(6,iN) = mean(timesFMF,'all');
    timeNs(7,iN) = mean(timesSMF,'all');

    % Display results
    tableRMSE = array2table(rmseNs,'RowNames',{'PMF','BMF','UMF','DMF','RMF','FMF','SMF'});
    tableNEES = array2table(neesNs,'RowNames',{'PMF','BMF','UMF','DMF','RMF','FMF','SMF'});
    format shortE
    clc; disp(tableRMSE); disp(tableNEES);
end

%% Grid Study

% RMSE
lw = 5;
ms = 12;
figure();
plot(Ns.^2,rmseNs(1,:),'-o','color',C(end,:),'linewidth',lw,'Marker','^','MarkerSize',ms,'DisplayName','PMF'); hold on;
plot(Ns.^2,rmseNs(2,:),'-o','color',C(end-3,:),'linewidth',lw,'Marker','+','MarkerSize',ms,'DisplayName','PMF-BER');
plot(Ns.^2,rmseNs(3,:),'-o','color',C(end-2,:),'linewidth',lw,'Marker','*','MarkerSize',ms,'DisplayName','PMF-UKF');
plot(Ns.^2,rmseNs(4,:),'-o','color',C(end-1,:),'linewidth',lw,'Marker','square','MarkerSize',ms,'DisplayName','PMF-DHF');
plot(Ns.^2,rmseNs(5,:),'-o','color',C(end-4,:),'linewidth',lw,'Marker','diamond','MarkerSize',ms,'DisplayName','FMF1');
plot(Ns.^2,rmseNs(6,:),'-o','color',C(end-5,:),'linewidth',lw,'Marker','x','MarkerSize',ms,'DisplayName','FMF2');
plot(Ns.^2,rmseNs(7,:),'-o','color',C(end-6,:),'linewidth',lw,'Marker','o','MarkerSize',ms,'DisplayName','SMF');
plot(Ns.^2,ones(length(Ns),1)*4.9649e-01,'--','color',[0.7 0.7 0.7],'linewidth',lw,'HandleVisibility','off');
text(Ns(6)^2,0.425,'SIR $\to \infty$','Interpreter','latex','FontSize',20,'color',[0.6 0.6 0.6]);
legend('Interpreter','Latex','FontSize',20,'NumColumns',1,'FontWeight','bold');
xlabel('Grid points','Interpreter','Latex','FontWeight','bold');
ylabel('RMSE','Interpreter','Latex','FontWeight','bold');
xlim([Ns(1)^2,Ns(end)^2]);
ylim([0.35,1.4]);
ytickformat('%.1f');
set(gca,'XMinorTick','on','YMinorTick','on');
set(gca,'LineWidth',4);
set(gca,'TickLabelInterpreter','latex','FontSize',20);

% SNEES
figure();
plot(Ns.^2,neesNs(1,:),'-o','color',C(end,:),'linewidth',lw,'Marker','^','MarkerSize',ms,'DisplayName','PMF'); hold on;
plot(Ns.^2,neesNs(2,:),'-o','color',C(end-3,:),'linewidth',lw,'Marker','+','MarkerSize',ms,'DisplayName','PMF-BER');
plot(Ns.^2,neesNs(3,:),'-o','color',C(end-2,:),'linewidth',lw,'Marker','*','MarkerSize',ms,'DisplayName','PMF-UKF');
plot(Ns.^2,neesNs(4,:),'-o','color',C(end-1,:),'linewidth',lw,'Marker','square','MarkerSize',ms,'DisplayName','PMF-DHF');
plot(Ns.^2,neesNs(5,:),'-o','color',C(end-4,:),'linewidth',lw,'Marker','diamond','MarkerSize',ms,'DisplayName','FMF1');
plot(Ns.^2,neesNs(6,:),'-o','color',C(end-5,:),'linewidth',lw,'Marker','x','MarkerSize',ms,'DisplayName','FMF2');
plot(Ns.^2,neesNs(7,:),'-o','color',C(end-6,:),'linewidth',lw,'Marker','o','MarkerSize',ms,'DisplayName','SMF');
plot(Ns.^2,ones(length(Ns),1),'--','color',[0.7 0.7 0.7],'linewidth',lw,'HandleVisibility','off');
text(Ns(2)^2,2,'SNEES = 1','Interpreter','latex','FontSize',20,'color',[0.6 0.6 0.6]);
legend('Interpreter','Latex','FontSize',20,'NumColumns',1,'FontWeight','bold');
xlabel('Grid points','Interpreter','Latex','FontWeight','bold');
ylabel('SNEES','Interpreter','Latex','FontWeight','bold');
set(gca,'XMinorTick','on','YMinorTick','on');
set(gca,'LineWidth',4);
xlim([Ns(1)^2,Ns(end)^2]);
ylim([0,20]);
ytickformat('%.0f');
set(gca,'TickLabelInterpreter','latex','FontSize',20);

% Time
figure();
semilogx(timeNs(1,:),rmseNs(1,:),'-o','color',C(end,:),'linewidth',lw,'Marker','^','MarkerSize',ms,'DisplayName','PMF'); hold on;
semilogx(timeNs(2,:),rmseNs(2,:),'-o','color',C(end-3,:),'linewidth',lw,'Marker','+','MarkerSize',ms,'DisplayName','PMF-BER');
semilogx(timeNs(3,:),rmseNs(3,:),'-o','color',C(end-2,:),'linewidth',lw,'Marker','*','MarkerSize',ms,'DisplayName','PMF-UKF');
semilogx(timeNs(4,:),rmseNs(4,:),'-o','color',C(end-1,:),'linewidth',lw,'Marker','square','MarkerSize',ms,'DisplayName','PMF-DHF');
semilogx(timeNs(5,:),rmseNs(5,:),'-o','color',C(end-4,:),'linewidth',lw,'Marker','diamond','MarkerSize',ms,'DisplayName','FMF1');
semilogx(timeNs(6,:),rmseNs(6,:),'-o','color',C(end-5,:),'linewidth',lw,'Marker','x','MarkerSize',ms,'DisplayName','FMF2');
semilogx(timeNs(7,:),rmseNs(7,:),'-o','color',C(end-6,:),'linewidth',lw,'Marker','o','MarkerSize',ms,'DisplayName','SMF');
semilogx(timeNs(1,:),ones(length(Ns),1)*4.9649e-01,'--','color',[0.7 0.7 0.7],'linewidth',lw,'HandleVisibility','off');
text(0.003,0.425,'SIR $\to \infty$','Interpreter','latex','FontSize',20,'color',[0.6 0.6 0.6]);
legend('Interpreter','Latex','FontSize',20,'NumColumns',2,'FontWeight','bold');
xlabel('Runtime (s)','Interpreter','Latex','FontWeight','bold');
ylabel('RMSE','Interpreter','Latex','FontWeight','bold');
ylim([0.35,1.4]);
xlim([min(timeNs(1,:)),max(timeNs(5,:))]);
ytickformat('%.1f');
xticks([0.001 0.002 0.005]);
set(gca,'XMinorTick','on','YMinorTick','on');
set(gca,'LineWidth',4);
set(gca,'TickLabelInterpreter','latex','FontSize',20);
