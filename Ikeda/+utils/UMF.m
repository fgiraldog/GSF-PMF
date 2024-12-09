function [Xhatk,whatk,xhatk,Phatk] = UMF(X,w,Q,N,sigG,y,R)

    %%% Propagation %%%
    invQ    = inv(Q);
    [s,n]   = size(X);
    whatkm1 = w;
    Xbark   = utils.f(X);
    xbark   = Xbark*whatkm1.';
    Pbark   = (Xbark - xbark)*diag(whatkm1)*(Xbark - xbark).' + Q;
    Pbark   = (Pbark + Pbark.')/2;

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
    YkUKF     = utils.h(XkUKF);
    weightsm  = [w0mu, wimu*ones(1, 2*(s))];
    ykUKF     = YkUKF*weightsm.';
    weightsc  = [w0cu, wimu*ones(1, 2*(s))];
    Pyyk      = (YkUKF - ykUKF)*diag(weightsc)*(YkUKF - ykUKF).' + R;
    Pxyk      = (XkUKF - xbark)*diag(weightsc)*(YkUKF - ykUKF).';
    xhatk     = xbark + Pxyk*(Pyyk\(y - ykUKF));
    Phatk     = Pbark - Pxyk*(Pyyk\(Pxyk.'));
    Phatk     = (Phatk + Phatk.')/2;
       
    %%% New grid %%%
    Xhatk    = utils.gridP(xhatk,Phatk,N,sigG);
    wbark    = zeros(1,n);
    whatk    = zeros(1,n);
    lwhatkm1 = log(whatkm1);
    for i = 1:n
        xi       = Xhatk(:,i);
        v        = xi - Xbark;
        v        = reshape(v,s,1,n);
        vt       = permute(v,[2 1 3]);
        wkj      = lwhatkm1 + reshape(-0.5*pagemtimes(vt,pagemtimes(invQ,v)),1,n);
        m        = max(wkj);
        wbark(i) = m + log(sum(exp(wkj - m)));
    end
    m       = max(wbark);
    wbark   = exp(wbark - (m + log(sum(exp(wbark - m)))));
    wbark   = wbark/sum(wbark);
    for i = 1:n
        nuyk     = y - utils.h(Xhatk(:,i));
        whatk(i) = log(wbark(i)) + (-0.5*nuyk.'*(R\nuyk));
    end
    m     = max(whatk);
    whatk = exp(whatk - (m + log(sum(exp(whatk - m)))));
    whatk = whatk/sum(whatk);
    xhatk = Xhatk*whatk.';
    Phatk = (Xhatk - xhatk)*diag(whatk)*(Xhatk - xhatk).';
    Phatk = (Phatk + Phatk.')/2;
    
end