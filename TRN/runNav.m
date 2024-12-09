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

%%%%%%%%%%%%%%%%%%% Maps for MCs %%%%%%%%%%%%%%%%%%%
% download map D02_028047_2037_dem.tif from: https://www.higp.hawaii.edu/prpdc/CTX_DEMs/Landforms/Nili
% store in +utils named as landingsite.tif
el        = im2double(imread('+utils/landingsite.tif')); 
elbase    = el(9000:9300,4000:4700);     % cropping the map so that the interpolator is not as big
elc       = elbase;
pixW      = 6;
r1        = 0:pixW:pixW*(size(elc,2)-1); % columns are x-direction
r2        = 0:pixW:pixW*(size(elc,1)-1); % rows are y-direction
sept      = 4;
elmt      = elc(1:sept:end,1:sept:end);  % smoothing the map
x1t       = r1(1:sept:end,1:sept:end);   % smoothing the map
x2t       = r2(1:sept:end,1:sept:end);   % smoothing the map

%% Define reference trajectory

%%%%%%%%%%%%%%%%%%% Reference trajectory %%%%%%%%%%%%%%%%%%%
r10       = 400;  % real starting position is + 3000 due to cropped map
r20       = 1500; % real starting position is + 2000 due to cropped map
theta0    = deg2rad(70);
v10       = 23*cos(theta0);
v20       = -23*sin(theta0);
r0        = [r10;r20];
v0        = [v10;v20];
x0        = [r0;v0];
nTime     = 200;
dt        = 1;
time      = 0:dt:nTime;
nSequence = length(time);
xs        = zeros(4,nSequence);
xk        = x0;
xs(:,1)   = x0;
turnr     = 2*theta0/nTime;
Sw        = 0.0001;
Fk        = utils.stm(turnr,dt);
Qk        = utils.QCT(Sw,dt,turnr);
for iSequence = 2:nSequence
    xk    = Fk*xk + sqrtm(Qk)*randn(4,1);
    xs(:,iSequence) = xk;
end

%%%%%%%%%%%%%%%%%%% Gridded Interpolants %%%%%%%%%%%%%%%%%%%
[R1,R2]   = ndgrid(x1t,x2t);
intt      = griddedInterpolant(R1,R2,elmt.','spline');
ys        = utils.h(xs,intt);

%%%%%%%%%%%%%%%%%%% Checking Trajectory %%%%%%%%%%%%%%%%%%%
figure()
plot(time,ys+1.7102e+02,'LineWidth',5,'Color',C(1,:));
xlim([0,nTime]);
ylabel('Elevation (m)','Interpreter','Latex','FontWeight','bold');
xlabel('Time (s)','Interpreter','Latex','FontWeight','bold');
set(gca,'XMinorTick','on','YMinorTick','on');
set(gca,'LineWidth',4)
set(gca,'TickLabelInterpreter','latex','FontSize',20);

%% Filtering

%%%%%%%%%%%%%%%%%%% Saving memory %%%%%%%%%%%%%%%%%%%
clearvars -except xs time x0 intt dt elmt x1t x2t Fk Qk Gk Sw C

%%%%%%%%%%%%%%%%%%% Problem Setup %%%%%%%%%%%%%%%%%%%
nMonte    = 1000;
nSequence = length(time);
nStates   = 4;
P0        = blkdiag(50,50,1,1);
R         = 0.1^2;
sqP0      = sqrtm(P0);  
sqR       = sqrt(R);
sigG      = 5;

%%%%%%%%%%%%%%%%%%% Preallocation %%%%%%%%%%%%%%%%%%%
Ns        = 16:4:36;
RrmseNs   = zeros([6,length(Ns)]);
VrmseNs   = zeros([6,length(Ns)]);
neesNs    = zeros([6,length(Ns)]);

for iN = 1:length(Ns)

    %%%%%%%%%%%%%%%%%%% Preallocation %%%%%%%%%%%%%%%%%%%
    % Truth
    xtruth    = zeros([nStates,nSequence,nMonte]);
    ywnoise   = zeros([1,nSequence,nMonte]);

    % PMF
    NPMF      = Ns(iN);
    xtildePMF = zeros([nStates,nSequence,nMonte]);
    PiiPMF    = zeros([nStates,nSequence,nMonte]);
    sneesPMF  = zeros([1,nSequence,nMonte]);

    % PMF-BER
    NBMF      = Ns(iN);
    xtildeBMF = zeros([nStates,nSequence,nMonte]);
    PiiBMF    = zeros([nStates,nSequence,nMonte]);
    sneesBMF  = zeros([1,nSequence,nMonte]);

    % PMF-UKF
    NUMF      = Ns(iN);
    xtildeUMF = zeros([nStates,nSequence,nMonte]);
    PiiUMF    = zeros([nStates,nSequence,nMonte]);
    sneesUMF  = zeros([1,nSequence,nMonte]);

    % FMF
    NRMF      = Ns(iN);
    xtildeRMF = zeros([nStates,nSequence,nMonte]);
    PiiRMF    = zeros([nStates,nSequence,nMonte]);
    sneesRMF  = zeros([1,nSequence,nMonte]);

    % SMF
    NSMF      = Ns(iN);
    xtildeSMF = zeros([nStates,nSequence,nMonte]);
    PiiSMF    = zeros([nStates,nSequence,nMonte]);
    sneesSMF  = zeros([1,nSequence,nMonte]);
    
    if iN == 1
        % EKF
        xtildeEKF = zeros([nStates,nSequence,nMonte]);
        PiiEKF    = zeros([nStates,nSequence,nMonte]);
        sneesEKF  = zeros([1,nSequence,nMonte]);
    end

    %%%%%%%%%%%%%%%%%%% Monte Carlo Loop %%%%%%%%%%%%%%%%%%%
    parfor iMonte = 1:nMonte; rng(iMonte + 777,'twister');

        %%%%%%%%%%%%%%%%%%% Initialization %%%%%%%%%%%%%%%%%%%
        % Truth
        xkm1        = 0;
        xk          = x0;
        xhatk       = xk + sqP0*randn(nStates,1);

        % PMF
        xhatkm1PMF  = 0;
        Phatkm1PMF  = 0;
        Rhatkm1PMF  = 0;
        xhatkPMF    = xhatk;
        PhatkPMF    = P0;
        RhatkPMF    = utils.gridP(xhatkPMF(1:2),PhatkPMF(1:2,1:2),NPMF,sigG);
        ngridPMF    = size(RhatkPMF,2);
        whatkm1PMF  = 0;
        whatkPMF    = zeros([1,ngridPMF]);
        for iPoint = 1:ngridPMF
            nuxk             = RhatkPMF(:,iPoint)-xhatkPMF(1:2);
            whatkPMF(iPoint) = (-0.5*nuxk.'*(PhatkPMF(1:2,1:2)\nuxk));
        end
        m           = max(whatkPMF);
        whatkPMF    = exp(whatkPMF - (m + log(sum(exp(whatkPMF - m)))));
        whatkPMF    = whatkPMF/sum(whatkPMF);

        % PMF-BER
        xhatkm1BMF  = 0;
        Phatkm1BMF  = 0;
        Rhatkm1BMF  = 0;
        xhatkBMF    = xhatk;
        PhatkBMF    = P0;
        RhatkBMF    = utils.gridP(xhatkBMF(1:2),PhatkBMF(1:2,1:2),NBMF,sigG);
        ngridBMF    = size(RhatkBMF,2);
        whatkm1BMF  = 0;
        whatkBMF    = zeros([1,ngridBMF]);
        for iPoint = 1:ngridBMF
            nuxk             = RhatkBMF(:,iPoint)-xhatkBMF(1:2);
            whatkBMF(iPoint) = (-0.5*nuxk.'*(PhatkBMF(1:2,1:2)\nuxk));
        end
        m           = max(whatkBMF);
        whatkBMF    = exp(whatkBMF - (m + log(sum(exp(whatkBMF - m)))));
        whatkBMF    = whatkBMF/sum(whatkBMF);

        % PMF-UKF
        xhatkm1UMF  = 0;
        Phatkm1UMF  = 0;
        Rhatkm1UMF  = 0;
        xhatkUMF    = xhatk;
        PhatkUMF    = P0;
        RhatkUMF    = utils.gridP(xhatkUMF(1:2),PhatkUMF(1:2,1:2),NUMF,sigG);
        ngridUMF    = size(RhatkUMF,2);
        whatkm1UMF  = 0;
        whatkUMF    = zeros([1,ngridUMF]);
        for iPoint = 1:ngridUMF
            nuxk             = RhatkUMF(:,iPoint)-xhatkUMF(1:2);
            whatkUMF(iPoint) = (-0.5*nuxk.'*(PhatkUMF(1:2,1:2)\nuxk));
        end
        m           = max(whatkUMF);
        whatkUMF    = exp(whatkUMF - (m + log(sum(exp(whatkUMF - m)))));
        whatkUMF    = whatkUMF/sum(whatkUMF);

        % FMF
        xhatkm1RMF  = 0;
        Phatkm1RMF  = 0;
        Rhatkm1RMF  = 0;
        xhatkRMF    = xhatk;
        PhatkRMF    = P0;
        RhatkRMF    = utils.gridP(xhatkRMF(1:2),PhatkRMF(1:2,1:2),NRMF,sigG);
        ngridRMF    = size(RhatkRMF,2);
        whatkm1RMF  = 0;
        whatkRMF    = zeros([1,ngridRMF]);
        for iPoint = 1:ngridRMF
            nuxk             = RhatkRMF(:,iPoint)-xhatkRMF(1:2);
            whatkRMF(iPoint) = (-0.5*nuxk.'*(PhatkRMF(1:2,1:2)\nuxk));
        end
        m           = max(whatkRMF);
        whatkRMF    = exp(whatkRMF - (m + log(sum(exp(whatkRMF - m)))));
        whatkRMF    = whatkRMF/sum(whatkRMF);

        % SMF
        xhatkm1SMF  = 0;
        Phatkm1SMF  = 0;
        Rhatkm1SMF  = 0;
        xhatkSMF    = xhatk;
        PhatkSMF    = P0;
        RhatkSMF    = utils.gridP(xhatkSMF(1:2),PhatkSMF(1:2,1:2),NSMF,sigG);
        ngridSMF    = size(RhatkSMF,2);
        whatkm1SMF  = 0;
        whatkSMF    = zeros([1,ngridSMF]);
        for iPoint = 1:ngridSMF
            nuxk             = RhatkSMF(:,iPoint)-xhatkSMF(1:2);
            whatkSMF(iPoint) = (-0.5*nuxk.'*(PhatkSMF(1:2,1:2)\nuxk));
        end
        m           = max(whatkSMF);
        whatkSMF    = exp(whatkSMF - (m + log(sum(exp(whatkSMF - m)))));
        whatkSMF    = whatkSMF/sum(whatkSMF);
        betaSMF     = 0.1;

        % EKF
        xhatkm1EKF  = 0;
        Phatkm1EKF  = 0;
        xhatkEKF    = xhatk;
        PhatkEKF    = P0;

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

                % FMF
                xtildekRMF                    = xk - xhatkRMF;
                xtildeRMF(:,iSequence,iMonte) = xtildekRMF;
                PiiRMF(:,iSequence,iMonte)    = diag(PhatkRMF);
                sneesRMF(:,iSequence,iMonte)  = xtildekRMF.'*pinv(PhatkRMF)*xtildekRMF;

                % SMF
                xtildekSMF                    = xk - xhatkSMF;
                xtildeSMF(:,iSequence,iMonte) = xtildekSMF;
                PiiSMF(:,iSequence,iMonte)    = diag(PhatkSMF);
                sneesSMF(:,iSequence,iMonte)  = xtildekSMF.'*pinv(PhatkSMF)*xtildekSMF;

                if iN == 1
                    % EKF
                    xtildekEKF                    = xk - xhatkEKF;
                    xtildeEKF(:,iSequence,iMonte) = xtildekEKF;
                    PiiEKF(:,iSequence,iMonte)    = diag(PhatkEKF);
                    sneesEKF(:,iSequence,iMonte)  = xtildekEKF.'*pinv(PhatkEKF)*xtildekEKF;
                end

                %%%%%%%%%%%%%%%%%%% Update time index %%%%%%%%%%%%%%%%%%%
                % Truth
                xkm1       = xk;

                % PMF
                xhatkm1PMF = xhatkPMF;
                Phatkm1PMF = PhatkPMF;
                Rhatkm1PMF = RhatkPMF;
                whatkm1PMF = whatkPMF;

                % PMF-BER
                xhatkm1BMF = xhatkBMF;
                Phatkm1BMF = PhatkBMF;
                Rhatkm1BMF = RhatkBMF;
                whatkm1BMF = whatkBMF;

                % PMF-UKF
                xhatkm1UMF = xhatkUMF;
                Phatkm1UMF = PhatkUMF;
                Rhatkm1UMF = RhatkUMF;
                whatkm1UMF = whatkUMF;

                % FMF
                xhatkm1RMF = xhatkRMF;
                Phatkm1RMF = PhatkRMF;
                Rhatkm1RMF = RhatkRMF;
                whatkm1RMF = whatkRMF;

                % SMF
                xhatkm1SMF = xhatkSMF;
                Phatkm1SMF = PhatkSMF;
                Rhatkm1SMF = RhatkSMF;
                whatkm1SMF = whatkSMF;

                if iN == 1
                    % EKF
                    xhatkm1EKF = xhatkEKF;
                    Phatkm1EKF = PhatkEKF;
                end

                continue
            end

            %%%%%%%%%%%%%%%%%% Truth %%%%%%%%%%%%%%%%%%%
            xk = Fk*xkm1 + sqrtm(Qk)*randn(4,1);
            yk = utils.h(xk(1:2),intt) + sqR*randn();

            %%%%%%%%%%%%%%%%%%% PMF %%%%%%%%%%%%%%%%%%%
            [RhatkPMF,whatkPMF,xhatkPMF,PhatkPMF] = utils.PMF(Rhatkm1PMF,whatkm1PMF,xhatkm1PMF,Phatkm1PMF,Fk,Qk,NPMF,sigG,yk,R,intt);

            %%%%%%%%%%%%%%%%%%% PMF-BER %%%%%%%%%%%%%%%%%%%
            [RhatkBMF,whatkBMF,xhatkBMF,PhatkBMF] = utils.BMF(Rhatkm1BMF,whatkm1BMF,xhatkm1BMF,Phatkm1BMF,Fk,Qk,NBMF,sigG,yk,R,intt);

            %%%%%%%%%%%%%%%%%%% PMF-UKF %%%%%%%%%%%%%%%%%%%
            [RhatkUMF,whatkUMF,xhatkUMF,PhatkUMF] = utils.UMF(Rhatkm1UMF,whatkm1UMF,xhatkm1UMF,Phatkm1UMF,Fk,Qk,NUMF,sigG,yk,R,intt);
            
            %%%%%%%%%%%%%%%%%%% FMF %%%%%%%%%%%%%%%%%%%
            [RhatkRMF,whatkRMF,xhatkRMF,PhatkRMF] = utils.RMF(Rhatkm1RMF,whatkm1RMF,xhatkm1RMF,Phatkm1RMF,Fk,Qk,NRMF,sigG,yk,R,intt);

            %%%%%%%%%%%%%%%%%% SMF %%%%%%%%%%%%%%%%%%%
            [RhatkSMF,whatkSMF,xhatkSMF,PhatkSMF] = utils.SMF(Rhatkm1SMF,whatkm1SMF,xhatkm1SMF,Phatkm1SMF,Fk,Qk,NSMF,sigG,yk,R,betaSMF,intt);

            if iN == 1
                %%%%%%%%%%%%%%%%%%% EKF %%%%%%%%%%%%%%%%%%%
                [xhatkEKF,PhatkEKF]               = utils.EKF(xhatkm1EKF,Phatkm1EKF,Fk,Qk,yk,R,intt);
            end

            %%%%%%%%%%%%%%%%%%% Saving metrics %%%%%%%%%%%%%%%%%%%
            % Truth
            xtruth(:,iSequence,iMonte)    = xk;
            ywnoise(:,iSequence,iMonte)   = yk;

            % PMF
            xtildekPMF                    = xk - xhatkPMF;
            xtildePMF(:,iSequence,iMonte) = xtildekPMF;
            PiiPMF(:,iSequence,iMonte)    = diag(PhatkPMF);
            sneeskPMF                     = xtildekPMF.'*pinv(PhatkPMF)*xtildekPMF;
            if abs(sneeskPMF) > 1e4; sneeskPMF = nan; end
            sneesPMF(:,iSequence,iMonte)  = sneeskPMF;

            % PMF-BER
            xtildekBMF                    = xk - xhatkBMF;
            xtildeBMF(:,iSequence,iMonte) = xtildekBMF;
            PiiBMF(:,iSequence,iMonte)    = diag(PhatkBMF);
            sneeskBMF                     = xtildekBMF.'*pinv(PhatkBMF)*xtildekBMF;
            if abs(sneeskBMF) > 1e4; sneeskBMF = nan; end
            sneesBMF(:,iSequence,iMonte)  = sneeskBMF;

            % PMF-UKF
            xtildekUMF                    = xk - xhatkUMF;
            xtildeUMF(:,iSequence,iMonte) = xtildekUMF;
            PiiUMF(:,iSequence,iMonte)    = diag(PhatkUMF);
            sneeskUMF                     = xtildekUMF.'*pinv(PhatkUMF)*xtildekUMF;
            if abs(sneeskUMF) > 1e4; sneeskUMF = nan; end
            sneesUMF(:,iSequence,iMonte)  = sneeskUMF;

            % FMF
            xtildekRMF                    = xk - xhatkRMF;
            xtildeRMF(:,iSequence,iMonte) = xtildekRMF;
            PiiRMF(:,iSequence,iMonte)    = diag(PhatkRMF);
            sneeskRMF                     = xtildekRMF.'*pinv(PhatkRMF)*xtildekRMF;
            if abs(sneeskRMF) > 1e4; sneeskRMF = nan; end
            sneesRMF(:,iSequence,iMonte)  = sneeskRMF;

            % SMF
            xtildekSMF                    = xk - xhatkSMF;
            xtildeSMF(:,iSequence,iMonte) = xtildekSMF;
            PiiSMF(:,iSequence,iMonte)    = diag(PhatkSMF);
            sneeskSMF                     = xtildekSMF.'*pinv(PhatkSMF)*xtildekSMF;
            if abs(sneeskSMF) > 1e4; sneeskSMF = nan; end
            sneesSMF(:,iSequence,iMonte)  = sneeskSMF;

            if iN == 1
                % EKF
                xtildekEKF                    = xk - xhatkEKF;
                xtildeEKF(:,iSequence,iMonte) = xtildekEKF;
                PiiEKF(:,iSequence,iMonte)    = diag(PhatkEKF);
                sneeskEKF                     = xtildekEKF.'*pinv(PhatkEKF)*xtildekEKF;
                if abs(sneeskEKF) > 1e4; sneeskEKF = nan; end
                sneesEKF(:,iSequence,iMonte)  = sneeskEKF;
            end

            %%%%%%%%%%%%%%%%%%% Update time index %%%%%%%%%%%%%%%%%%%
            % Truth
            xkm1       = xk;

            % PMF
            xhatkm1PMF = xhatkPMF;
            Phatkm1PMF = PhatkPMF;
            Rhatkm1PMF = RhatkPMF;
            whatkm1PMF = whatkPMF;

            % PMF-BER
            xhatkm1BMF = xhatkBMF;
            Phatkm1BMF = PhatkBMF;
            Rhatkm1BMF = RhatkBMF;
            whatkm1BMF = whatkBMF;

            % PMF-UKF
            xhatkm1UMF = xhatkUMF;
            Phatkm1UMF = PhatkUMF;
            Rhatkm1UMF = RhatkUMF;
            whatkm1UMF = whatkUMF;

            % FMF
            xhatkm1RMF = xhatkRMF;
            Phatkm1RMF = PhatkRMF;
            Rhatkm1RMF = RhatkRMF;
            whatkm1RMF = whatkRMF;

            % SMF
            xhatkm1SMF = xhatkSMF;
            Phatkm1SMF = PhatkSMF;
            Rhatkm1SMF = RhatkSMF;
            whatkm1SMF = whatkSMF;

            if iN == 1
                % EKF
                xhatkm1EKF = xhatkEKF;
                Phatkm1EKF = PhatkEKF;
            end
        end
    end
   

    % RMSE
    RrmsePMF = mean(mean(sqrt((mean(xtildePMF(1:2,:,:).^2,1))),3));
    RrmseBMF = mean(mean(sqrt((mean(xtildeBMF(1:2,:,:).^2,1))),3));
    RrmseUMF = mean(mean(sqrt((mean(xtildeUMF(1:2,:,:).^2,1))),3));
    RrmseRMF = mean(mean(sqrt((mean(xtildeRMF(1:2,:,:).^2,1))),3));
    RrmseSMF = mean(mean(sqrt((mean(xtildeSMF(1:2,:,:).^2,1))),3));
    if iN == 1
        RrmseEKF = mean(mean(sqrt((mean(xtildeEKF(1:2,:,:).^2,1))),3));
        RrmseNs(:,iN) = [RrmsePMF;RrmseBMF;RrmseUMF;RrmseRMF;RrmseSMF;RrmseEKF];
    else
        RrmseNs(:,iN) = [RrmsePMF;RrmseBMF;RrmseUMF;RrmseRMF;RrmseSMF;nan];
    end

    VrmsePMF = mean(mean(sqrt((mean(xtildePMF(3:4,:,:).^2,1))),3));
    VrmseBMF = mean(mean(sqrt((mean(xtildeBMF(3:4,:,:).^2,1))),3));
    VrmseUMF = mean(mean(sqrt((mean(xtildeUMF(3:4,:,:).^2,1))),3));
    VrmseRMF = mean(mean(sqrt((mean(xtildeRMF(3:4,:,:).^2,1))),3));
    VrmseSMF = mean(mean(sqrt((mean(xtildeSMF(3:4,:,:).^2,1))),3));
    if iN == 1
        VrmseEKF = mean(mean(sqrt((mean(xtildeEKF(3:4,:,:).^2,1))),3));
        VrmseNs(:,iN) = [VrmsePMF;VrmseBMF;VrmseUMF;VrmseRMF;VrmseSMF;VrmseEKF];
    else
        VrmseNs(:,iN) = [VrmsePMF;VrmseBMF;VrmseUMF;VrmseRMF;VrmseSMF;nan];
    end

    % SNEES
    neesPMF = mean(sneesPMF,'all','omitnan')./nStates;
    neesBMF = mean(sneesBMF,'all','omitnan')./nStates;
    neesUMF = mean(sneesUMF,'all','omitnan')./nStates;
    neesRMF = mean(sneesRMF,'all','omitnan')./nStates;
    neesSMF = mean(sneesSMF,'all','omitnan')./nStates;
    if iN == 1
        neesEKF = mean(sneesEKF,'all','omitnan')./nStates;
        neesNs(:,iN) = [neesPMF;neesBMF;neesUMF;neesRMF;neesSMF;neesEKF];
    else
        neesNs(:,iN) = [neesPMF;neesBMF;neesUMF;neesRMF;neesSMF;nan];
    end

    % Display results
    tableRRMSE = array2table(RrmseNs,'RowNames',{'PMF','BMF','UMF','FMF','SMF','EKF'});
    tableVRMSE = array2table(VrmseNs,'RowNames',{'PMF','BMF','UMF','FMF','SMF','EKF'});
    tableNEES  = array2table(neesNs,'RowNames',{'PMF','BMF','UMF','FMF','SMF','EKF'});
    format shortE
    clc; disp(tableRRMSE); disp(tableVRMSE); disp(tableNEES);
end

%% Grid Study

% Position RMSE
lw = 5;
ms = 12;
figure();
plot(Ns.^2,RrmseNs(1,:),'-o','color',C(end,:),'linewidth',lw,'Marker','^','MarkerSize',ms,'DisplayName','PMF'); hold on;
plot(Ns.^2,RrmseNs(2,:),'-o','color',C(end-3,:),'linewidth',lw,'Marker','+','MarkerSize',ms,'DisplayName','PMF-BER');
plot(Ns.^2,RrmseNs(3,:),'-o','color',C(end-2,:),'linewidth',lw,'Marker','*','MarkerSize',ms,'DisplayName','PMF-UKF');
plot(Ns.^2,RrmseNs(4,:),'-o','color',C(end-4,:),'linewidth',lw,'Marker','diamond','MarkerSize',ms,'DisplayName','FMF1');
plot(Ns.^2,RrmseNs(5,:),'-o','color',C(end-6,:),'linewidth',lw,'Marker','o','MarkerSize',ms,'DisplayName','SMF');
plot(Ns.^2,ones(length(Ns),1)*RrmseNs(end,1),'--','color',[0.7 0.7 0.7],'linewidth',lw,'HandleVisibility','off');
text(Ns(2)^2,RrmseNs(end,1) + 0.5,'EKF','Interpreter','latex','FontSize',20,'color',[0.6 0.6 0.6]);
legend('Interpreter','Latex','FontSize',20,'NumColumns',1,'FontWeight','bold');
xlabel('Grid points','Interpreter','Latex','FontWeight','bold');
ylabel('\it Position \rm RMSE (m)','Interpreter','Latex','FontWeight','bold');
% ylim([0,15])
xlim([250,Ns(end)^2]);
xticks([250 450 650 850 1050 1250])
set(gca,'XMinorTick','on','YMinorTick','on');
set(gca,'LineWidth',4);
set(gca,'TickLabelInterpreter','latex','FontSize',20);

% Velocity RMSE
figure();
plot(Ns.^2,VrmseNs(1,:),'-o','color',C(end,:),'linewidth',lw,'Marker','^','MarkerSize',ms,'DisplayName','PMF'); hold on;
plot(Ns.^2,VrmseNs(2,:),'-o','color',C(end-3,:),'linewidth',lw,'Marker','+','MarkerSize',ms,'DisplayName','PMF-BER');
plot(Ns.^2,VrmseNs(3,:),'-o','color',C(end-2,:),'linewidth',lw,'Marker','*','MarkerSize',ms,'DisplayName','PMF-UKF');
plot(Ns.^2,VrmseNs(4,:),'-o','color',C(end-4,:),'linewidth',lw,'Marker','diamond','MarkerSize',ms,'DisplayName','FMF1');
plot(Ns.^2,VrmseNs(5,:),'-o','color',C(end-6,:),'linewidth',lw,'Marker','o','MarkerSize',ms,'DisplayName','SMF');
plot(Ns.^2,ones(length(Ns),1)*VrmseNs(end,1),'--','color',[0.7 0.7 0.7],'linewidth',lw,'HandleVisibility','off');
text(Ns(2)^2,VrmseNs(end,1) + 0.015,'EKF','Interpreter','latex','FontSize',20,'color',[0.6 0.6 0.6]);
legend('Interpreter','Latex','FontSize',20,'NumColumns',1,'FontWeight','bold');
xlabel('Grid points','Interpreter','Latex','FontWeight','bold');
ylabel('\it Velocity \rm RMSE (m/s)','Interpreter','Latex','FontWeight','bold');
% ylim([0 0.4])
xlim([250,Ns(end)^2]);
xticks([250 450 650 850 1050 1250])
set(gca,'XMinorTick','on','YMinorTick','on');
set(gca,'LineWidth',4);
set(gca,'TickLabelInterpreter','latex','FontSize',20);

% SNEES
figure();
plot(Ns.^2,neesNs(1,:),'-o','color',C(end,:),'linewidth',lw,'Marker','^','MarkerSize',ms,'DisplayName','PMF'); hold on;
plot(Ns.^2,neesNs(2,:),'-o','color',C(end-3,:),'linewidth',lw,'Marker','+','MarkerSize',ms,'DisplayName','PMF-BER');
plot(Ns.^2,neesNs(3,:),'-o','color',C(end-2,:),'linewidth',lw,'Marker','*','MarkerSize',ms,'DisplayName','PMF-UKF');
plot(Ns.^2,neesNs(4,:),'-o','color',C(end-4,:),'linewidth',lw,'Marker','diamond','MarkerSize',ms,'DisplayName','FMF1');
plot(Ns.^2,neesNs(5,:),'-o','color',C(end-6,:),'linewidth',lw,'Marker','o','MarkerSize',ms,'DisplayName','SMF');
plot(Ns.^2,ones(length(Ns),1),'--','color',[0.7 0.7 0.7],'linewidth',lw,'HandleVisibility','off');
text(Ns(2)^2 - 100,3,'SNEES = 1','Interpreter','latex','FontSize',20,'color',[0.6 0.6 0.6]);
legend('Interpreter','Latex','FontSize',20,'NumColumns',1,'FontWeight','bold');
xlabel('Grid points','Interpreter','Latex','FontWeight','bold');
ylabel('SNEES','Interpreter','Latex','FontWeight','bold');
set(gca,'XMinorTick','on','YMinorTick','on');
set(gca,'LineWidth',4);
ylim([0,30]);
xlim([250,Ns(end)^2]);
xticks([250 450 650 850 1050 1250])
ytickformat('%.0f');
set(gca,'TickLabelInterpreter','latex','FontSize',20);