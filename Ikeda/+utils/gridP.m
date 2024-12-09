function G = gridP(mean,cov,N,sigG)

    [T,L]   = eig(cov);
    L       = diag(L);
    gLim    = sqrt(L)*sigG;
    [~,I]   = sort(diag(cov));
    [~,I]   = sort(I);
    [l,I2]  = sort(gLim);
    gLim    = l(I);
    l2      = T(:,I2);
    T       = l2(:,I);
    Xs      = -gLim(1):2*gLim(1)/(N-1):gLim(1);
    Ys      = -gLim(2):2*gLim(2)/(N-1):gLim(2);
    XY      = combvec(Xs,Ys);
    rotXY   = T*XY;
    G       = rotXY + mean;
    
end