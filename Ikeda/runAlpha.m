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

%%%%%%%%%%%%%%%%%%% Alpha Study Setup %%%%%%%%%%%%%%%%%%%
As     = 0:0.05:0.75;
rmseNs = zeros([2,length(As)]);
neesNs = zeros([2,length(As)]);
sigG   = 3;

for iA = 1:length(As)
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

    % Fusion Mass Filter 1
    NRMF      = 19;
    xtildeRMF = zeros([nStates,nSequence,nMonte]);
    PiiRMF    = zeros([nStates,nSequence,nMonte]);
    sneesRMF  = zeros([1,nSequence,nMonte]);
    timesRMF  = zeros([1,nSequence,nMonte]);
   
    % Silverman Mass Filter
    NSMF      = 19;
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
        betaSMF     = As(iA);

        %%%%%%%%%%%%%%%%%%% Time Loop %%%%%%%%%%%%%%%%%%%
        for iSequence = 1:nSequence

            if iSequence == 1

                %%%%%%%%%%%%%%%%%%% Saving metrics %%%%%%%%%%%%%%%%%%%
                % Truth
                xtruth(:,iSequence,iMonte)    = xk;

                % FMF1
                xtildekRMF                    = xk - xhatkRMF;
                xtildeRMF(:,iSequence,iMonte) = xtildekRMF;
                PiiRMF(:,iSequence,iMonte)    = diag(PhatkRMF);
                sneesRMF(:,iSequence,iMonte)  = xtildekRMF.'*pinv(PhatkRMF)*xtildekRMF;

                % SMF
                xtildekSMF                    = xk - xhatkSMF;
                xtildeSMF(:,iSequence,iMonte) = xtildekSMF;
                PiiSMF(:,iSequence,iMonte)    = diag(PhatkSMF);
                sneesSMF(:,iSequence,iMonte)  = xtildekSMF.'*pinv(PhatkSMF)*xtildekSMF;
            
                %%%%%%%%%%%%%%%%%%% Update time index %%%%%%%%%%%%%%%%%%%
                % Truth
                xkm1  = xk;

                % FMF1
                Xhatkm1RMF = XhatkRMF;
                whatkm1RMF = whatkRMF;

                % SMF
                Xhatkm1SMF = XhatkSMF;
                whatkm1SMF = whatkSMF;   

                continue
            end

            %%%%%%%%%%%%%%%%%% Truth %%%%%%%%%%%%%%%%%%%
            xk = utils.f(xkm1) + sqQt*randn([nStates,1]);
            yk = utils.h(xk) + sqR*randn();
            
            if iA == 1
                %%%%%%%%%%%%%%%%%%% FMF1 %%%%%%%%%%%%%%%%%%%
                [XhatkRMF,whatkRMF,xhatkRMF,PhatkRMF] = utils.RMF(Xhatkm1RMF,whatkm1RMF,Qf,NRMF,sigG,yk,R);
            end

            %%%%%%%%%%%%%%%%%%% SMF %%%%%%%%%%%%%%%%%%%
            [XhatkSMF,whatkSMF,xhatkSMF,PhatkSMF] = utils.SMF(Xhatkm1SMF,whatkm1SMF,Qf,NSMF,sigG,yk,R,betaSMF);

            %%%%%%%%%%%%%%%%%%% Saving metrics %%%%%%%%%%%%%%%%%%%
            % Truth
            xtruth(:,iSequence,iMonte)    = xk;

            % FMF1
            xtildekRMF                    = xk - xhatkRMF;
            xtildeRMF(:,iSequence,iMonte) = xtildekRMF;
            PiiRMF(:,iSequence,iMonte)    = diag(PhatkRMF);
            sneeskRMF                     = xtildekRMF.'*pinv(PhatkRMF)*xtildekRMF;
            if sneeskRMF > 1e4;sneeskRMF  = nan; end
            sneesRMF(:,iSequence,iMonte)  = sneeskRMF;

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

            % FMF1
            Xhatkm1RMF = XhatkRMF;
            whatkm1RMF = whatkRMF;

            % SMF
            Xhatkm1SMF = XhatkSMF;
            whatkm1SMF = whatkSMF;
        end
    end

    % RMSE
    rmseNs(1,iA) = mean(mean(sqrt((mean(xtildeRMF.^2,1))),3));
    rmseNs(2,iA) = mean(mean(sqrt((mean(xtildeSMF.^2,1))),3));

    % SNEES
    neesNs(1,iA) = mean(sneesRMF,'all','omitnan')./nStates;
    neesNs(2,iA) = mean(sneesSMF,'all','omitnan')./nStates;

    % Display results
    tableRMSE = array2table(rmseNs,'RowNames',{'RMF','SMF'});
    tableNEES = array2table(neesNs,'RowNames',{'RMF','SMF'});
    format shortE
    clc; disp(tableRMSE); disp(tableNEES);
end

%% Alpha Study

% RMSE
lw = 5;
ms = 12;
figure();
colororder(C(1:2:3,:));
yyaxis right;
plot(As,rmseNs(1,1)*ones(1,length(As)),':','linewidth',lw,'DisplayName','FMF1'); hold on;
plot(As,rmseNs(2,:),'-','linewidth',lw,'Marker','o','MarkerSize',ms,'DisplayName','SMF');
xlabel('$\alpha$','Interpreter','Latex','FontWeight','bold');
ylabel('RMSE','Interpreter','Latex','FontWeight','bold');
xlim([As(1),As(end)]);
ylim([0.513 0.530]);
ytickformat('%.3f');
set(gca,'XMinorTick','on','YMinorTick','on');
set(gca,'LineWidth',4);
set(gca,'TickLabelInterpreter','latex','FontSize',20);

% SNEES
yyaxis left;
plot(As,ones(1,length(As)),'-.','Color',C(1,:),'linewidth',lw/2,'HandleVisibility','off');
plot(As,neesNs(1,1)*ones(1,length(As)),':','linewidth',lw,'DisplayName','FMF1');
plot(As,neesNs(2,:),'-','linewidth',lw,'Marker','o','MarkerSize',ms,'DisplayName','SMF');
text(0.75/2,1.02,'SNEES = 1','Interpreter','latex','FontSize',20,'Color',C(1,:));
legend('Interpreter','Latex','FontSize',20,'NumColumns',4,'FontWeight','bold','Location','south');
xlabel('$\alpha$','Interpreter','Latex','FontWeight','bold');
ylabel('SNEES','Interpreter','Latex','FontWeight','bold');
set(gca,'XMinorTick','on','YMinorTick','on');
set(gca,'LineWidth',4);
ylim([0.7 1.2]);
xlim([As(1),As(end)]);
ytickformat('%.1f');
set(gca,'TickLabelInterpreter','latex','FontSize',20);

