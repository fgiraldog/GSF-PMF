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

rng(0); % Reproducibility of results

%%%%%%%%%%%%%%%%%%% Problem Setup %%%%%%%%%%%%%%%%%%%
N       = 25;
sigG    = 3;
xbark   = [-3.5; 0];
Pbark   = [1,1/2; 1/2,1];
y       = 1;
R       = 0.1^2;
Q       = 2e-1*eye(2);
invQ    = inv(Q);
s       = 2;
n       = N^2;
Xbark   = mvnrnd(xbark,Pbark,n).';
whatkm1 = ones(1,n)/n;
whatkm1 = whatkm1/sum(whatkm1);

%%%%%%%%%%%%%%%%%%% Problem Settings %%%%%%%%%%%%%%%%%%%
t           = tiledlayout(2,3,'Padding','compact','TileSpacing','compact');
sizemax     = 100;
sizemin     = 4;
gauss       = @(x) exp(-0.5*sum((x - xbark).*(Pbark\(x-xbark)),1));
obs         = @(x) exp(-0.5*sum((h(x) - y).*(R\(h(x)-y)),1));
NPlot       = 200;
xs          = linspace(-8,2,NPlot);
ys          = linspace(-4.5,4.5,NPlot);
[X,Y]       = meshgrid(xs,ys);
XY          = [X(:),Y(:)].';
priorXY     = reshape(gauss(XY),NPlot,[]);
levelsPrior = max(priorXY(:)).*[normpdf(3),normpdf(2),normpdf(1)]/normpdf(0);
obsXY       = reshape(obs(XY),NPlot,[]);
levelsObs   = max(obsXY(:)).*[normpdf(3),normpdf(2)]/normpdf(0);
postXY      = reshape(obs(XY).*gauss(XY),NPlot,[]);
levelsPost  = max(postXY(:)).*[normpdf(3),normpdf(2),normpdf(1)]/normpdf(0);

%% PMF

%%% New grid %%%
Xk      = utils.gridP(xbark,Pbark,N,sigG);
wbark   = zeros(1,n);
for i = 1:n
    xi       = Xk(:,i);
    v        = xi - Xbark;
    v        = reshape(v,s,1,n);
    vt       = permute(v,[2 1 3 ]);
    wkj      = log(whatkm1) + reshape(-0.5*pagemtimes(vt,pagemtimes(invQ,v)),1,n);
    m        = max(wkj);
    wbark(i) = m + log(sum(exp(wkj - m)));
end
m       = max(wbark);
wbark   = exp(wbark - (m + log(sum(exp(wbark - m)))));
wbark   = wbark/sum(wbark);

%%% Update %%%
Xhatk   = Xk;
whatk   = zeros(1,n);
for i = 1:n
    nuyk     = y - h(Xhatk(:,i));
    whatk(i) = log(wbark(i)) + (-0.5*nuyk.'*(R\nuyk));
end
m      = max(whatk);
whatk  = exp(whatk - (m + log(sum(exp(whatk - m)))));
whatk  = whatk/sum(whatk);
sizess = (whatk/max(whatk))*(sizemax - sizemin) + sizemin;

%%% Plotting %%%
nexttile(1);
contour(xs,ys,priorXY,levelsPrior,'-','EdgeColor',C(1,:),'LineWidth',4,'DisplayName','Prior'); hold on;
contour(xs,ys,obsXY,levelsObs,'-','EdgeColor',C(2,:),'LineWidth',4,'DisplayName','Measurement');
contour(xs,ys,postXY,levelsPost,'-','EdgeColor',C(5,:),'LineWidth',4,'DisplayName','Posterior');
scatter(Xhatk(1,:),Xhatk(2,:),sizess,C(3,:),'filled','HandleVisibility','off'); hold off;
ylabel('$x_2$','Interpreter','Latex');
axis equal;
xlim([-7.25,1.5]);
ylim([-3,3]);
set(gca,'LineWidth',4);
set(gca,'xticklabel',[]);
set(gca,'TickLabelInterpreter','latex','FontSize',25);
title('PMF','Interpreter','latex');

%% PMF-BER

clearvars -except C N sigG xbark Pbark y R gauss obs NPlot xs ys X Y XY t Q invQ s n Xbark whatkm1 sizemax sizemin priorXY levelsPrior obsXY levelsObs postXY levelsPost

%%% New grid %%%
Xk      = utils.gridP(xbark,Pbark,N,sigG);
wbark   = zeros(1,n);
for i = 1:n
    xi       = Xk(:,i);
    v        = xi - Xbark;
    v        = reshape(v,s,1,n);
    vt       = permute(v,[2 1 3]);
    wkj      = log(whatkm1) + reshape(-0.5*pagemtimes(vt,pagemtimes(invQ,v)),1,n);
    m        = max(wkj);
    wbark(i) = m + log(sum(exp(wkj - m)));
end
m       = max(wbark);
wbark   = exp(wbark - (m + log(sum(exp(wbark - m)))));
wbark   = wbark/sum(wbark);

%%% Update %%%
Xhatk   = Xk;
whatk   = zeros(1,n);
for i = 1:n
    nuyk     = y - h(Xhatk(:,i));
    whatk(i) = log(wbark(i)) + (-0.5*nuyk.'*(R\nuyk));
end
m     = max(whatk);
whatk = exp(whatk - (m + log(sum(exp(whatk - m)))));
whatk = whatk/sum(whatk);
xhatk = Xhatk*whatk.';
Phatk = (Xhatk - xhatk)*diag(whatk)*(Xhatk - xhatk).';
Phatk = (Phatk + Phatk.')/2;

%%% New grid %%%
Xhatk   = utils.gridP(xhatk,Phatk,N,sigG);
wbark   = zeros(1,n);
whatk   = zeros(1,n);
for i = 1:n
    xi       = Xhatk(:,i);
    v        = xi - Xbark;
    v        = reshape(v,s,1,n);
    vt       = permute(v,[2 1 3]);
    wkj      = log(whatkm1) + reshape(-0.5*pagemtimes(vt,pagemtimes(invQ,v)),1,n);
    m        = max(wkj);
    wbark(i) = m + log(sum(exp(wkj - m)));
end
m       = max(wbark);
wbark   = exp(wbark - (m + log(sum(exp(wbark - m)))));
wbark   = wbark/sum(wbark);
for i = 1:n
    nuyk     = y - h(Xhatk(:,i));
    whatk(i) = log(wbark(i)) + (-0.5*nuyk.'*(R\nuyk));
end
m      = max(whatk);
whatk  = exp(whatk - (m + log(sum(exp(whatk - m)))));
whatk  = whatk/sum(whatk);
sizess = (whatk/max(whatk))*(sizemax - sizemin) + sizemin;

%%% Plotting %%%
nexttile(2);
contour(xs,ys,priorXY,levelsPrior,'-','EdgeColor',C(1,:),'LineWidth',4,'DisplayName','Prior'); hold on;
contour(xs,ys,obsXY,levelsObs,'-','EdgeColor',C(2,:),'LineWidth',4,'DisplayName','Measurement')
contour(xs,ys,postXY,levelsPost,'-','EdgeColor',C(5,:),'LineWidth',4,'DisplayName','Posterior');
scatter(Xhatk(1,:),Xhatk(2,:),sizess,C(3,:),'filled','HandleVisibility','off'); hold off;
axis equal
set(gca,'yticklabel',[])
set(gca,'xticklabel',[])
xlim([-7.25,1.5])
ylim([-3,3])
set(gca,'LineWidth',4)
set(gca,'TickLabelInterpreter','latex','FontSize',25);
title('PMF-BER','Interpreter','latex');

%% PMF-UKF

clearvars -except C N sigG xbark Pbark y R gauss obs NPlot xs ys X Y XY t Q invQ s n Xbark whatkm1 sizemax sizemin priorXY levelsPrior obsXY levelsObs postXY levelsPost

%%% Update %%%
alpha     = 1;
beta      = 2;
kappa     = 3 - (s);
lambdau   = alpha^2*(s + kappa) - (s);
w0mu      = lambdau/(s + lambdau);
wimu      = 1/(2*(s + lambdau));
w0cu      = lambdau/(s + lambdau) + (1 - alpha^2 + beta);
Sbark     = chol(Pbark).';
nSbark    = sqrt(s + lambdau)*Sbark;
XkUKF     = [xbark, repmat(xbark,1,s) + nSbark, repmat(xbark,1,s) - nSbark];
YkUKF     = h(XkUKF);
weightsm  = [w0mu, wimu*ones(1, 2*(s))];
ykUKF     = YkUKF*weightsm.';
weightsc  = [w0cu, wimu*ones(1, 2*(s))];
Pyyk      = (YkUKF - ykUKF)*diag(weightsc)*(YkUKF - ykUKF).' + R;
Pxyk      = (XkUKF - xbark)*diag(weightsc)*(YkUKF - ykUKF).';
xhatk     = xbark + Pxyk*(Pyyk\(y - ykUKF));
Phatk     = Pbark - Pxyk*(Pyyk\(Pxyk.'));
Phatk     = (Phatk + Phatk.')/2;
   
%%% New grid %%%
Xhatk   = utils.gridP(xhatk,Phatk,N,sigG);
wbark   = zeros(1,n);
whatk   = zeros(1,n);
for i = 1:n
    xi       = Xhatk(:,i);
    v        = xi - Xbark;
    v        = reshape(v,s,1,n);
    vt       = permute(v,[2 1 3]);
    wkj      = log(whatkm1) + reshape(-0.5*pagemtimes(vt,pagemtimes(invQ,v)),1,n);
    m        = max(wkj);
    wbark(i) = m + log(sum(exp(wkj - m)));
end
m       = max(wbark);
wbark   = exp(wbark - (m + log(sum(exp(wbark - m)))));
wbark   = wbark/sum(wbark);
for i = 1:n
    nuyk     = y - h(Xhatk(:,i));
    whatk(i) = log(wbark(i)) + (-0.5*nuyk.'*(R\nuyk));
end
m      = max(whatk);
whatk  = exp(whatk - (m + log(sum(exp(whatk - m)))));
whatk  = whatk/sum(whatk);
sizess = (whatk/max(whatk))*(sizemax - sizemin) + sizemin;

%%% Plotting %%%
nexttile(3);
contour(xs,ys,priorXY,levelsPrior,'-','EdgeColor',C(1,:),'LineWidth',4,'HandleVisibility','off'); hold on;
contour(xs,ys,obsXY,levelsObs,'-','EdgeColor',C(2,:),'LineWidth',4,'HandleVisibility','off')
contour(xs,ys,postXY,levelsPost,'-','EdgeColor',C(5,:),'LineWidth',4,'DisplayName','Posterior');
scatter(Xhatk(1,:),Xhatk(2,:),sizess,C(3,:),'filled','HandleVisibility','off'); hold off;
axis equal;
axis tight;
xlim([-7.25,1.5])
ylim([-3,3])
set(gca,'LineWidth',4)
set(gca,'yticklabel',[])
set(gca,'xticklabel',[])
set(gca,'TickLabelInterpreter','latex','FontSize',25);
title('PMF-UKF','Interpreter','latex');

%% PMF-DHF 

clearvars -except C N sigG xbark Pbark y R gauss obs NPlot xs ys X Y XY t Q invQ s n Xbark whatkm1 sizemax sizemin priorXY levelsPrior obsXY levelsObs postXY levelsPost

%%% Update %%%
XB      = utils.gridP(xbark,Pbark,N,sigG);
hullIdx = boundary(XB.',1);
XB      = XB(:,hullIdx);
[~,nB]  = size(XB);
nLambda = 25;
dLambda = 1/nLambda;
for iLambda = 1:nLambda
    kLambda = iLambda*dLambda;
    Hk      = reshape(Hh(XB),1,2,[]); 
    Ht      = permute(Hk,[2 1 3]);
    A       = -0.5.*pagemtimes(Pbark,Ht);
    A       = pagemrdivide(A,kLambda*pagemtimes(pagemtimes(Hk,Pbark),Ht)+ R);
    A       = pagemtimes(A,Hk);
    v       = reshape(y - h(XB),1,1,nB) + pagemtimes(Hk,reshape(XB,s,1,nB));
    b       = eye(2,2) + kLambda*A;
    b       = pagemtimes(b,pagemrdivide(pagemtimes(Pbark,Ht),R));
    b       = pagemtimes(b,v);
    b       = b + pagemtimes(A,xbark);
    b       = pagemtimes(eye(2,2) + 2*kLambda*A,b);
    fk      = pagemtimes(A,reshape(XB,s,1,[])) + b;
    XB      = XB + reshape(fk,s,nB)*dLambda;
end

%%% New grid %%%
Xhatk   = utils.gridD(XB,N);
wbark   = zeros(1,n);
whatk   = zeros(1,n);
for i = 1:n
    xi       = Xhatk(:,i);
    v        = xi - Xbark;
    v        = reshape(v,s,1,n);
    vt       = permute(v,[2 1 3]);
    wkj      = log(whatkm1) + reshape(-0.5*pagemtimes(vt,pagemtimes(invQ,v)),1,n);
    m        = max(wkj);
    wbark(i) = m + log(sum(exp(wkj - m)));
end
m       = max(wbark);
wbark   = exp(wbark - (m + log(sum(exp(wbark - m)))));
wbark   = wbark/sum(wbark);
for i = 1:n
    nuyk     = y - h(Xhatk(:,i));
    whatk(i) = log(wbark(i)) + (-0.5*nuyk.'*(R\nuyk));
end
m      = max(whatk);
whatk  = exp(whatk - (m + log(sum(exp(whatk - m)))));
whatk  = whatk/sum(whatk);
sizess = (whatk/max(whatk))*(sizemax - sizemin) + sizemin;

%%% Plotting %%%
nexttile(4);
contour(xs,ys,priorXY,levelsPrior,'-','EdgeColor',C(1,:),'LineWidth',4,'HandleVisibility','off'); hold on;
contour(xs,ys,obsXY,levelsObs,'-','EdgeColor',C(2,:),'LineWidth',4,'HandleVisibility','off')
contour(xs,ys,postXY,levelsPost,'-','EdgeColor',C(5,:),'LineWidth',4,'DisplayName','Posterior');
scatter(Xhatk(1,:),Xhatk(2,:),sizess,C(3,:),'filled','HandleVisibility','off'); hold off;
axis equal;
axis tight;
xlabel('$x_1$','interpreter','latex')
ylabel('$x_2$','interpreter','latex')
xlim([-7.25,1.5])
ylim([-3,3])
set(gca,'LineWidth',4)
set(gca,'TickLabelInterpreter','latex','FontSize',25);
title('PMF-DHF','Interpreter','latex');

%% FMF

clearvars -except C N sigG xbark Pbark y R gauss obs NPlot xs ys X Y XY t Q invQ s n Xbark whatkm1 sizemax sizemin priorXY levelsPrior obsXY levelsObs postXY levelsPost

%%% Update %%%
ny      = size(R,1);
wbark   = whatkm1;
eye_s   = eye(s);
H       = Hh(Xbark);
Ht      = permute(H,[2 1 3]);
W       = pagemtimes(pagemtimes(H,Q),Ht) + R;
K       = pagemrdivide(pagemtimes(Q,Ht),W);
v       = y - h(Xbark);
v       = reshape(v,ny,1,n);
vt      = permute(v,[2 1 3]);
XkGSF   = Xbark + reshape(pagemtimes(K,v),s,n);
KH      = pagemtimes(K,H);
Kt      = permute(K,[2 1 3]);
PkGSF   = pagemtimes(pagemtimes(eye_s - KH,Q),permute(eye_s - KH,[2 1 3])) + pagemtimes(pagemtimes(K,R),Kt);
wkGSF   = log(wbark) - log(sqrt(reshape(prod(pageeig(W,'vector'),1),1,n))) + reshape(-0.5*pagemtimes(vt,pagemldivide(W,v)),1,n);
m       = max(wkGSF);
wkGSF   = exp(wkGSF - (m + log(sum(exp(wkGSF - m)))));
wkGSF   = wkGSF/sum(wkGSF);
xhatk   = XkGSF*wkGSF.';
Phatk   = sum(PkGSF.*reshape(wkGSF,1,1,[]),3);
nuxk    = (XkGSF - xhatk);
Phatk   = Phatk + nuxk*diag(wkGSF)*nuxk.';
Phatk   = (Phatk + Phatk.')/2;

%%% New grid %%%
Xhatk   = utils.gridP(xhatk,Phatk,N,sigG);
wbark   = zeros(1,n);
whatk   = zeros(1,n);
for i = 1:n
    xi       = Xhatk(:,i);
    v        = xi - Xbark;
    v        = reshape(v,s,1,n);
    vt       = permute(v,[2 1 3]);
    wkj      = log(whatkm1) + reshape(-0.5*pagemtimes(vt,pagemtimes(invQ,v)),1,n);
    m        = max(wkj);
    wbark(i) = m + log(sum(exp(wkj - m)));
end
m       = max(wbark);
wbark   = exp(wbark - (m + log(sum(exp(wbark - m)))));
wbark   = wbark/sum(wbark);
for i = 1:n
    nuyk     = y - h(Xhatk(:,i));
    whatk(i) = log(wbark(i)) + (-0.5*nuyk.'*(R\nuyk));
end
m      = max(whatk);
whatk  = exp(whatk - (m + log(sum(exp(whatk - m)))));
whatk  = whatk/sum(whatk);
sizess = (whatk/max(whatk))*(sizemax - sizemin) + sizemin;

%%% Plotting %%%
nexttile(5);
contour(xs,ys,priorXY,levelsPrior,'-','EdgeColor',C(1,:),'LineWidth',4,'HandleVisibility','off'); hold on;
contour(xs,ys,obsXY,levelsObs,'-','EdgeColor',C(2,:),'LineWidth',4,'HandleVisibility','off')
contour(xs,ys,postXY,levelsPost,'-','EdgeColor',C(5,:),'LineWidth',4,'DisplayName','Posterior');
scatter(Xhatk(1,:),Xhatk(2,:),sizess,C(3,:),'filled','HandleVisibility','off'); hold off;
axis equal;
axis tight;
xlabel('$x_1$','interpreter','latex')
xlim([-7.25,1.5])
ylim([-3,3])
set(gca,'LineWidth',4)
set(gca,'yticklabel',[])
set(gca,'TickLabelInterpreter','latex','FontSize',25);
title('FMF','Interpreter','latex');

%% SMF

clearvars -except C N sigG xbark Pbark y R gauss obs NPlot xs ys X Y XY t Q invQ s n Xbark whatkm1 sizemax sizemin priorXY levelsPrior obsXY levelsObs postXY levelsPost

%%% Update %%%
ny      = size(R,1);
wbark   = whatkm1;
eye_s   = eye(s);
Ps      = 1*(4/((n)*(s+2)))^(2/(s+4))*Pbark + Q;
Ps      = (Ps + Ps.')/2;
invPs   = inv(Ps);
H       = Hh(Xbark);
Ht      = permute(H,[2 1 3]);
W       = pagemtimes(pagemtimes(H,Ps),Ht) + R;
K       = pagemrdivide(pagemtimes(Ps,Ht),W);
v       = y - h(Xbark); 
v       = reshape(v,ny,1,n);
vt      = permute(v,[2 1 3]);
XkGSF   = Xbark + reshape(pagemtimes(K,v),s,n);
KH      = pagemtimes(K,H);
Kt      = permute(K,[2 1 3]);
PkGSF   = pagemtimes(pagemtimes(eye_s - KH,Ps),permute(eye_s - KH,[2 1 3])) + pagemtimes(pagemtimes(K,R),Kt);
wkGSF   = log(wbark) - log(sqrt(reshape(prod(pageeig(W,'vector'),1),1,n))) + reshape(-0.5*pagemtimes(vt,pagemldivide(W,v)),1,n);
m       = max(wkGSF);
wkGSF   = exp(wkGSF - (m + log(sum(exp(wkGSF - m)))));
wkGSF   = wkGSF/sum(wkGSF);
xhatk   = XkGSF*wkGSF.';
Phatk   = sum(PkGSF.*reshape(wkGSF,1,1,[]),3);
nuxk    = (XkGSF - xhatk);
Phatk   = Phatk + nuxk*diag(wkGSF)*nuxk.';
Phatk   = (Phatk + Phatk.')/2;

%%% New grid %%%
Xhatk   = utils.gridP(xhatk,Phatk,N,sigG);
wbark   = zeros(1,n);
whatk   = zeros(1,n);
for i = 1:n
    xi       = Xhatk(:,i);
    v        = xi - Xbark;
    v        = reshape(v,s,1,n) ;
    vt       = permute(v,[2 1 3]);
    wkj      = log(whatkm1) + reshape(-0.5*pagemtimes(vt,pagemtimes(invPs,v)),1,n);
    m        = max(wkj);
    wbark(i) = m + log(sum(exp(wkj - m)));
end
m       = max(wbark);
wbark   = exp(wbark - (m + log(sum(exp(wbark - m)))));
wbark   = wbark/sum(wbark);
for i = 1:n
    nuyk     = y - h(Xhatk(:,i));
    whatk(i) = log(wbark(i)) + (-0.5*nuyk.'*(R\nuyk));
end
m       = max(whatk);
whatk   = exp(whatk - (m + log(sum(exp(whatk - m)))));
whatk   = whatk/sum(whatk);
sizess  = (whatk/max(whatk))*(sizemax - sizemin) + sizemin;

%%% Plotting %%%
nexttile(6);
contour(xs,ys,priorXY,levelsPrior,'-','EdgeColor',C(1,:),'LineWidth',4,'DisplayName','Prior'); hold on;
contour(xs,ys,obsXY,levelsObs,'-','EdgeColor',C(2,:),'LineWidth',4,'DisplayName','Measurement');
contour(xs,ys,postXY,levelsPost,'-','EdgeColor',C(5,:),'LineWidth',4,'DisplayName','Posterior');
scatter(Xhatk(1,:),Xhatk(2,:),sizess,C(3,:),'filled','HandleVisibility','off'); hold off;
axis equal;
axis tight;
xlabel('$x_1$','interpreter','latex')
xlim([-7.25,1.5])
ylim([-3,3])
set(gca,'LineWidth',4)
set(gca,'yticklabel',[])
set(gca,'TickLabelInterpreter','latex','FontSize',25);
title('SMF','Interpreter','latex');
lg = legend('Interpreter','Latex','FontSize',25,'NumColumns',3,'Location','southoutside');
lg.Layout.Tile = 'south';

%% Helper functions

function y = h(x)
    y = sqrt(x(1,:).^2 + x(2,:).^2);
end

function J = Hh(x)
    x = x.';
    J = [x(:,1)./sqrt(x(:,1).^2+x(:,2).^2),x(:,2)./sqrt(x(:,1).^2+x(:,2).^2)];
    J = reshape(J.',1,2,[]);
end
