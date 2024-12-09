function [Rhatk,whatk,xhatk,Phatk] = UMF(Rhatkm1,whatkm1,xhatkm1,Phatkm1,Fk,Qk,N,sigG,yk,R,int)

    %%% Propagation %%%
    [s,n]    = size(Rhatkm1);
    Vhatkm1  = utils.gridP(xhatkm1(3:4),Phatkm1(3:4,3:4),N,sigG);
    Xhatkm1  = [Rhatkm1;Vhatkm1];
    Xbark    = Fk*Xhatkm1;
    wbark    = whatkm1;
    vbark    = mean(Xbark(3:4,:),2);
    Rbark    = Xbark(1:2,:);
    rbark    = Rbark*wbark.';
    Pbark    = (Rbark - rbark)*diag(whatkm1)*(Rbark - rbark).';
    Pbark    = (Pbark + Pbark.')/2;
    PbarkEKF = Fk*Phatkm1*Fk.' + Qk;
    PbarkEKF = (PbarkEKF + PbarkEKF.')/2;
    
    %%% Update %%%
    alpha     = 1;
    beta      = 2;
    kappa     = 3 - (s);
    lambdau   = alpha^2*(s + kappa) - (s);
    w0mu      = lambdau/(s + lambdau);
    wimu      = 1/(2*(s + lambdau));
    w0cu      = lambdau/(s + lambdau) + (1 - alpha^2 + beta);
    Fv        = Fk(1:2,3:4);
    Qv        = Fv*Phatkm1(3:4,3:4)*Fv.';
    Qv        = (Qv + Qv.')/2;
    Ps        = Qk(1:2,1:2) + Qv;
    Ps        = (Ps + Ps.')/2;
    invPs     = inv(Ps);
    Sbark     = chol(Pbark + Ps).';
    nSbark    = sqrt(s + lambdau)*Sbark;
    RkUKF     = [rbark, repmat(rbark,1,s) + nSbark, repmat(rbark,1,s) - nSbark];
    YkUKF     = utils.h(RkUKF,int);
    weightsm  = [w0mu, wimu*ones(1, 2*(s))];
    ykUKF     = YkUKF*weightsm.';
    weightsc  = [w0cu, wimu*ones(1, 2*(s))];
    Pyyk      = (YkUKF - ykUKF)*diag(weightsc)*(YkUKF - ykUKF).' + R;
    Pxyk      = (RkUKF - rbark)*diag(weightsc)*(YkUKF - ykUKF).';
    rhatk     = rbark + Pxyk*(Pyyk\(yk - ykUKF));
    Phatk     = (Pbark + Ps) - Pxyk*(Pyyk\(Pxyk.'));
    Phatk     = (Phatk + Phatk.')/2;

    %%% New grid %%%
    Rhatk    = utils.gridP(rhatk,Phatk,N,sigG);
    whatk    = zeros(1,n);
    lwhatkm1 = log(whatkm1);
    for i = 1:n
        ri       = Rhatk(:,i);
        v        = ri - Rbark;
        v        = reshape(v,s,1,n);
        vt       = permute(v,[2 1 3]);
        wkj      = lwhatkm1 + reshape(-0.5*pagemtimes(vt,pagemtimes(invPs,v)),1,n);
        m        = max(wkj);
        wbark(i) = m + log(sum(exp(wkj - m)));
    end
    m        = max(wbark);
    wbark    = exp(wbark - (m + log(sum(exp(wbark - m)))));
    wbark    = wbark/sum(wbark);
    Yk       = utils.h(Rhatk,int);
    for i = 1:n
        nuyk     = yk - Yk(i);
        whatk(i) = log(wbark(i)) + (-0.5*(nuyk.'/R)*nuyk);
    end
    m        = max(whatk);
    whatk    = exp(whatk - (m + log(sum(exp(whatk - m)))));
    whatk    = whatk/sum(whatk);
    rhatk    = Rhatk*whatk.';
    Phatk    = (Rhatk - rhatk)*diag(whatk)*(Rhatk - rhatk).';
    Phatk    = (Phatk + Phatk.')/2;
    vhatk    = vbark + (PbarkEKF(3:4,1:2)/PbarkEKF(1:2,1:2))*(rhatk - rbark);
    Pcrossk  = PbarkEKF(3:4,1:2)*(PbarkEKF(1:2,1:2)\Phatk);
    Pvhatk   = PbarkEKF(3:4,3:4) + PbarkEKF(3:4,1:2)*((PbarkEKF(1:2,1:2)\Phatk)/PbarkEKF(1:2,1:2) - PbarkEKF(1:2,1:2)^(-1))*PbarkEKF(1:2,3:4);
    xhatk    = [rhatk;vhatk];
    Phatk    = [Phatk Pcrossk.';Pcrossk Pvhatk];
    Phatk    = (Phatk + Phatk.')/2;

end