function [Xhatk,whatk,xhatk,Phatk] = DWF(X,w,Q,N,sigG,y,R,analytic)

    %%% Propagation %%%
    [s,n]   = size(X);
    whatkm1 = w;
    Xbark   = utils.f(X);
    xbark   = Xbark*whatkm1.';
    Pbark   = (Xbark - xbark)*diag(whatkm1)*(Xbark - xbark).' + Q;
    Pbark   = (Pbark + Pbark.')/2;

    %%% New grid %%%
    [Xk,deltai]  = utils.gridW(xbark,Pbark,N,sigG);
    wbark        = zeros(1,n);
    lwhatkm1     = log(whatkm1);
    if analytic
        cnorm   = 1/sqrt(det(Q)*((2*pi)^s));
        for j = 1:n
            xhatkj = Xk(:,j);
            I23j   = zeros(1,n);            
            for i = 1:n
                xhatkm1i  = X(:,i);
                lower     = xhatkm1i - deltai/2;
                upper     = xhatkm1i + deltai/2;
                F         = Fjacob(xhatkm1i(1),xhatkm1i(2));
                Finv      = pinv(F + eps(1));         % Avoids ill-conditioning
                Qi        = Finv*Q*Finv.';
                cauxi     = 1/sqrt(det(Qi)*((2*pi)^s));
                integral  = mvncdf(lower,upper,Finv*(xhatkj - Xbark(:,i) + F*xhatkm1i),Qi);
                I23j(:,i) = cnorm*integral/cauxi;
            end
            wkj      = lwhatkm1 + log(I23j + eps(1)); % Avoids log(0)
            m        = max(wkj);
            wbark(j) = m + log(sum(exp(wkj - m)));
        end
    else
        for j = 1:n
            xhatkj = Xk(:,j);
            I23j   = zeros(1,n);
            for i = 1:n
                xhatkm1i  = X(:,i);
                lower     = xhatkm1i - deltai/2;
                upper     = xhatkm1i + deltai/2;
                integrand = @(x,y) fint(x,y,xhatkj,Q);
                integral  = integral2(integrand,lower(1),upper(1),lower(2),upper(2));
                I23j(:,i) = integral;
            end
            wkj      = lwhatkm1 + log(I23j + eps(1)); % Avoids log(0)
            m        = max(wkj);
            wbark(j) = m + log(sum(exp(wkj - m)));
        end
    end
    m       = max(wbark);
    wbark   = exp(wbark - (m + log(sum(exp(wbark - m)))));
    wbark   = wbark/sum(wbark);  

    %%% Update %%%
    Xhatk   = Xk;
    whatk   = zeros(1,n);
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


function integrand = fint(x,y,xi,Q)

    [n1,n2]   = size(x);
    x         = x(:).';
    y         = y(:).';
    a         = 0.4;
    mu        = 0.9;
    b         = 6;
    A         = mu.*[cos(a - b./(1+(x.^2 + y.^2))); -sin(a - b./(1+(x.^2 + y.^2))); sin(a - b./(1+(x.^2 + y.^2))); cos(a - b./(1+(x.^2 + y.^2)))];
    A         = reshape(A,2,2,[]);
    xk        = [1;0] + reshape(pagemtimes(A,reshape([x;y],2,1,[])),2,[]);
    integrand = mvnpdf((xi - xk).',zeros(1,2),Q);
    integrand = reshape(integrand,n1,n2);

end

function jacobian = Fjacob(x,y)

    jacobian = reshape([cos(6.0./(x.^2+y.^2+1.0)-2.0./5.0).*(9.0./1.0e+1)+x.^2.*sin(6.0./(x.^2+y.^2+1.0)-2.0./5.0).*1.0./(x.^2+y.^2+1.0).^2.*(5.4e+1./5.0)+x.*y.*cos(6.0./(x.^2+y.^2+1.0)-2.0./5.0).*1.0./(x.^2+y.^2+1.0).^2.*(5.4e+1./5.0),sin(6.0./(x.^2+y.^2+1.0)-2.0./5.0).*(9.0./1.0e+1)-x.^2.*cos(6.0./(x.^2+y.^2+1.0)-2.0./5.0).*1.0./(x.^2+y.^2+1.0).^2.*(5.4e+1./5.0)+x.*y.*sin(6.0./(x.^2+y.^2+1.0)-2.0./5.0).*1.0./(x.^2+y.^2+1.0).^2.*(5.4e+1./5.0),sin(6.0./(x.^2+y.^2+1.0)-2.0./5.0).*(-9.0./1.0e+1)+y.^2.*cos(6.0./(x.^2+y.^2+1.0)-2.0./5.0).*1.0./(x.^2+y.^2+1.0).^2.*(5.4e+1./5.0)+x.*y.*sin(6.0./(x.^2+y.^2+1.0)-2.0./5.0).*1.0./(x.^2+y.^2+1.0).^2.*(5.4e+1./5.0),cos(6.0./(x.^2+y.^2+1.0)-2.0./5.0).*(9.0./1.0e+1)+y.^2.*sin(6.0./(x.^2+y.^2+1.0)-2.0./5.0).*1.0./(x.^2+y.^2+1.0).^2.*(5.4e+1./5.0)-x.*y.*cos(6.0./(x.^2+y.^2+1.0)-2.0./5.0).*1.0./(x.^2+y.^2+1.0).^2.*(5.4e+1./5.0)],2,2);

end