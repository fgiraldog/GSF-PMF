function J = Hh(x)

    x = x.';
    J = [x(:,1)./sqrt(x(:,1).^2+x(:,2).^2),x(:,2)./sqrt(x(:,1).^2+x(:,2).^2)];
    J = reshape(J.',1,2,[]);
    
end