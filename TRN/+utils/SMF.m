function [Rhatk,whatk,xhatk,Phatk] = SMF(Rhatkm1,whatkm1,xhatkm1,Phatkm1,Fk,Qk,N,sigG,yk,R,betaS,int)

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
    ys       = size(R,1);
    eyes     = eye(s);
    Fv       = Fk(1:2,3:4);
    Qv       = Fv*Phatkm1(3:4,3:4)*Fv.';
    Qv       = (Qv + Qv.')/2;
    Ps       = betaS*(4/((n)*(s+2)))^(2/(s+4))*Pbark + Qk(1:2,1:2) + Qv;
    Ps       = (Ps + Ps.')/2;
    invPs    = inv(Ps);  
    Hk       = utils.Hh(Rbark,int);
    Hkt      = permute(Hk,[2 1 3]);
    Wk       = pagemtimes(pagemtimes(Hk,Ps),Hkt) + R;
    Kk       = pagemrdivide(pagemtimes(Ps,Hkt),Wk);
    Kkt      = permute(Kk,[2 1 3]);
    nuk      = yk - utils.h(Rbark,int);
    nuk      = reshape(nuk,ys,1,n);
    nukt     = permute(nuk,[2 1 3]);
    RkGSF    = Rbark + reshape(pagemtimes(Kk,nuk),s,n);
    KHk      = pagemtimes(Kk,Hk);
    PkGSF    = pagemtimes(pagemtimes(eyes - KHk,Ps),permute(eyes - KHk,[2 1 3])) + pagemtimes(pagemtimes(Kk,R),Kkt);
    wkGSF    = log(wbark) - log(sqrt(reshape(prod(pageeig(Wk,'vector'),1),1,n))) + reshape(-0.5*pagemtimes(nukt,pagemldivide(Wk,nuk)),1,n);
    m        = max(wkGSF);
    wkGSF    = exp(wkGSF - (m + log(sum(exp(wkGSF - m)))));
    wkGSF    = wkGSF/sum(wkGSF);
    rhatk    = RkGSF*wkGSF.';
    Phatk    = sum(PkGSF.*reshape(wkGSF,1,1,[]),3);
    Phatk    = Phatk + (RkGSF - rhatk)*diag(wkGSF)*(RkGSF - rhatk).';
    Phatk    = (Phatk + Phatk.')/2;
    
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