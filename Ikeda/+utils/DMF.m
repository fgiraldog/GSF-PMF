function [Xhatk,whatk,xhatk,Phatk] = DMF(X,w,Q,N,sigG,y,R)

    %%% Propagation %%%
    invQ    = inv(Q);
    [s,n]   = size(X);
    whatkm1 = w;
    Xbark   = utils.f(X);
    xbark   = Xbark*whatkm1.';
    Pbark   = (Xbark - xbark)*diag(whatkm1)*(Xbark - xbark).' + Q;
    Pbark   = (Pbark + Pbark.')/2;
    
    %%% Update %%%
    XB      = utils.gridP(xbark,Pbark,N,sigG);
    hullIdx = boundary(XB.',1);
    XB      = XB(:,hullIdx);
    [~,nB]  = size(XB);
    nLambda = 25;
    dLambda = 1/nLambda;
    for iLambda = 1:nLambda
        kLambda = iLambda*dLambda;
        Hk      = reshape(utils.Hh(XB),1,2,[]); 
        Ht      = permute(Hk,[2 1 3]);
        A       = -0.5.*pagemtimes(Pbark,Ht);
        A       = pagemrdivide(A,kLambda*pagemtimes(pagemtimes(Hk,Pbark),Ht)+ R);
        A       = pagemtimes(A,Hk);
        v       = reshape(y - utils.h(XB),1,1,nB) + pagemtimes(Hk,reshape(XB,s,1,nB));
        b       = eye(2,2) + kLambda*A;
        b       = pagemtimes(b,pagemrdivide(pagemtimes(Pbark,Ht),R));
        b       = pagemtimes(b,v);
        b       = b + pagemtimes(A,xbark);
        b       = pagemtimes(eye(2,2) + 2*kLambda*A,b);
        fk      = pagemtimes(A,reshape(XB,s,1,[])) + b;
        XB      = XB + reshape(fk,s,nB)*dLambda;
    end

    %%% New grid %%%
    Xhatk    = utils.gridD(XB,N);
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
