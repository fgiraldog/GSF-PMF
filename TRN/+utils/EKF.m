function [xhatk,Phatk] = EKF(xhatkm1,Phatkm1,Fk,Qk,yk,R,int)
    
    %%% Propagation %%%
    xbark  = Fk*xhatkm1;
    Pbark  = Fk*Phatkm1*Fk.' + Qk;

    %%% Update %%%
    Hk     = [utils.Hh(xbark(1:2),int),0,0];
    nuk    = yk - utils.h(xbark(1:2),int);
    Pxyk   = Pbark*Hk.';
    Pyyk   = Hk*Pbark*Hk.' + R;
    Kk     = Pxyk/Pyyk;
    xhatk  = xbark + Kk*nuk;
    Phatk  = (eye(4)-Kk*Hk)*Pbark*(eye(4)-Kk*Hk).' + Kk*R*Kk.';
    
end