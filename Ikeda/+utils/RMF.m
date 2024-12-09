function [Xhatk,whatk,xhatk,Phatk] = RMF(X,w,Q,N,sigG,y,R)

    %%% Propagation %%%
    invQ    = inv(Q);
    [s,n]   = size(X);
    whatkm1 = w;
    Xbark   = utils.f(X);

    %%% Update %%%
    ys      = size(R,1);
    wbark   = whatkm1;
    eye_s   = eye(s);
    H       = utils.Hh(Xbark);
    Ht      = permute(H,[2 1 3]);
    W       = pagemtimes(pagemtimes(H,Q),Ht) + R;
    K       = pagemrdivide(pagemtimes(Q,Ht),W);
    v       = y - utils.h(Xbark);
    v       = reshape(v,ys,1,n);
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