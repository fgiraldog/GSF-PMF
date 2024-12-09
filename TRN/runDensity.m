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
Ns        = 16;
RrmseNs   = zeros([2,1]);
VrmseNs   = zeros([2,1]);
neesNs    = zeros([2,1]);
timeNs    = zeros([2,1]);

%%%%%%%%%%%%%%%%%%% Preallocation %%%%%%%%%%%%%%%%%%%
% Truth
xtruth    = zeros([nStates,nSequence,nMonte]);

% PMF-DWCa
NDWA      = Ns;
xtildeDWA = zeros([nStates,nSequence,nMonte]);
PiiDWA    = zeros([nStates,nSequence,nMonte]);
sneesDWA  = zeros([1,nSequence,nMonte]);
timesDWA  = zeros([1,nSequence,nMonte]);

% SMF
NSMF      = Ns;
xtildeSMF = zeros([nStates,nSequence,nMonte]);
PiiSMF    = zeros([nStates,nSequence,nMonte]);
sneesSMF  = zeros([1,nSequence,nMonte]);
timesSMF  = zeros([1,nSequence,nMonte]);


%%%%%%%%%%%%%%%%%%% Monte Carlo Loop %%%%%%%%%%%%%%%%%%%
parfor iMonte = 1:nMonte; rng(iMonte + 777,'twister');

    %%%%%%%%%%%%%%%%%%% Initialization %%%%%%%%%%%%%%%%%%%
    % Truth
    xkm1        = 0;
    xk          = x0;
    xhatk       = xk + sqP0*randn(nStates,1);

    % PMF-DWCa
    xhatkm1DWA  = 0;
    Phatkm1DWA  = 0;
    Rhatkm1DWA  = 0;
    xhatkDWA    = xhatk;
    PhatkDWA    = P0;
    RhatkDWA    = utils.gridP(xhatkDWA(1:2),PhatkDWA(1:2,1:2),NDWA,sigG);
    ngridDWA    = size(RhatkDWA,2);
    whatkm1DWA  = 0;
    whatkDWA    = zeros([1,ngridDWA]);
    for iPoint = 1:ngridDWA
        nuxk             = RhatkDWA(:,iPoint)-xhatkDWA(1:2);
        whatkDWA(iPoint) = (-0.5*nuxk.'*(PhatkDWA(1:2,1:2)\nuxk));
    end
    m           = max(whatkDWA);
    whatkDWA    = exp(whatkDWA - (m + log(sum(exp(whatkDWA - m)))));
    whatkDWA    = whatkDWA/sum(whatkDWA);
    
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
    betaSMF     = 0.3;

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

            % SMF
            xtildekSMF                    = xk - xhatkSMF;
            xtildeSMF(:,iSequence,iMonte) = xtildekSMF;
            PiiSMF(:,iSequence,iMonte)    = diag(PhatkSMF);
            sneesSMF(:,iSequence,iMonte)  = xtildekSMF.'*pinv(PhatkSMF)*xtildekSMF;

            %%%%%%%%%%%%%%%%%%% Update time index %%%%%%%%%%%%%%%%%%%
            % Truth
            xkm1       = xk;

            % PMF-DWCa
            xhatkm1DWA = xhatkDWA;
            Phatkm1DWA = PhatkDWA;
            Rhatkm1DWA = RhatkDWA;
            whatkm1DWA = whatkDWA;

            % SMF
            xhatkm1SMF = xhatkSMF;
            Phatkm1SMF = PhatkSMF;
            Rhatkm1SMF = RhatkSMF;
            whatkm1SMF = whatkSMF;

            continue
        end

        %%%%%%%%%%%%%%%%%% Truth %%%%%%%%%%%%%%%%%%%
        xk = Fk*xkm1 + sqrtm(Qk)*randn(4,1);
        yk = utils.h(xk(1:2),intt) + sqR*randn();

        %%%%%%%%%%%%%%%%%%% PMF-DWCa %%%%%%%%%%%%%%%%%%%
        tic;
        [RhatkDWA,whatkDWA,xhatkDWA,PhatkDWA] = utils.DWF(Rhatkm1DWA,whatkm1DWA,xhatkm1DWA,Phatkm1DWA,Fk,Qk,NDWA,sigG,yk,R,intt);
        timesDWA(:,iSequence,iMonte) = toc;

        %%%%%%%%%%%%%%%%%% SMF %%%%%%%%%%%%%%%%%%%
        tic;
        [RhatkSMF,whatkSMF,xhatkSMF,PhatkSMF] = utils.SMS(Rhatkm1SMF,whatkm1SMF,xhatkm1SMF,Phatkm1SMF,Fk,Qk,NSMF,sigG,yk,R,betaSMF,intt);
        timesSMF(:,iSequence,iMonte) = toc;

        %%%%%%%%%%%%%%%%%%% Saving metrics %%%%%%%%%%%%%%%%%%%
        % Truth
        xtruth(:,iSequence,iMonte)    = xk;

        % PMF-DWCa
        xtildekDWA                    = xk - xhatkDWA;
        xtildeDWA(:,iSequence,iMonte) = xtildekDWA;
        PiiDWA(:,iSequence,iMonte)    = diag(PhatkDWA);
        sneeskDWA                     = xtildekDWA.'*pinv(PhatkDWA)*xtildekDWA;
        if abs(sneeskDWA) > 1e4; sneeskDWA = nan; end
        sneesDWA(:,iSequence,iMonte)  = sneeskDWA;

        % SMF
        xtildekSMF                    = xk - xhatkSMF;
        xtildeSMF(:,iSequence,iMonte) = xtildekSMF;
        PiiSMF(:,iSequence,iMonte)    = diag(PhatkSMF);
        sneeskSMF                     = xtildekSMF.'*pinv(PhatkSMF)*xtildekSMF;
        if abs(sneeskSMF) > 1e4; sneeskSMF = nan; end
        sneesSMF(:,iSequence,iMonte)  = sneeskSMF;

        %%%%%%%%%%%%%%%%%%% Update time index %%%%%%%%%%%%%%%%%%%
        % Truth
        xkm1       = xk;

        % PMF-DWCa
        xhatkm1DWA = xhatkDWA;
        Phatkm1DWA = PhatkDWA;
        Rhatkm1DWA = RhatkDWA;
        whatkm1DWA = whatkDWA;

        % SMF
        xhatkm1SMF = xhatkSMF;
        Phatkm1SMF = PhatkSMF;
        Rhatkm1SMF = RhatkSMF;
        whatkm1SMF = whatkSMF;

    end
end

% RMSE
RrmseNs(1) = mean(mean(sqrt((mean(xtildeDWA(1:2,:,:).^2,1))),3));
RrmseNs(2) = mean(mean(sqrt((mean(xtildeSMF(1:2,:,:).^2,1))),3));

VrmseNs(1) = mean(mean(sqrt((mean(xtildeDWA(3:4,:,:).^2,1))),3));
VrmseNs(2) = mean(mean(sqrt((mean(xtildeSMF(3:4,:,:).^2,1))),3));

% SNEES
neesNs(1)  = mean(sneesDWA,'all','omitnan')./nStates;
neesNs(2)  = mean(sneesSMF,'all','omitnan')./nStates;

% Time
timeNs(1) = mean(timesDWA,'all');
timeNs(2) = mean(timesSMF,'all');

% Display results
tableRes   = array2table([RrmseNs VrmseNs neesNs timeNs],'RowNames',{'DWA','SMF'},'VariableNames',{'rRMSE','vRMSE','SNEES','Time'});
format shortE
clc; disp(tableRes);