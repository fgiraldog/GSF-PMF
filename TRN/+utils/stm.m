function Fmatrix = stm(w,dt)

    swt     = sin(w*dt);
    cwt     = cos(w*dt);
    Fmatrix = [1 0 swt/w     -(1-cwt)/w;
               0 1 (1-cwt)/w swt/w;
               0 0 cwt       -swt;
               0 0 swt        cwt];
end