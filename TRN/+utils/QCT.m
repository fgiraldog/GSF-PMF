function Qk = QCT(Sw,dt,w)

    wtswt = w*dt - sin(w*dt);
    eycwt = 1 - cos(w*dt);
    Qk    = Sw*[2*wtswt/w^3 0           eycwt/w^2  wtswt/w^2;
             0           2*wtswt/w^3 -wtswt/w^2 eycwt/w^2;
             eycwt/w^2   -wtswt/w^2  dt          0;
             wtswt/w^2   eycwt/w^2   0          dt];
    
end