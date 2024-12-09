function xk = f(xkm1)

    a         = 0.4;
    mu        = 0.9;
    b         = 6;
    A         = mu.*[cos(a - b./(1+(xkm1(1,:).^2 + xkm1(2,:).^2))); -sin(a - b./(1+(xkm1(1,:).^2 + xkm1(2,:).^2))); sin(a - b./(1+(xkm1(1,:).^2 + xkm1(2,:).^2))); cos(a - b./(1+(xkm1(1,:).^2 + xkm1(2,:).^2)))];
    A         = reshape(A,2,2,[]);
    xk        = [1;0] + reshape(pagemtimes(A,reshape(xkm1,2,1,[])),2,[]);
    
end