function G = gridD(XY,N)

    XY      = minBoundingBox(XY);
    meanG   = mean(XY,2);
    T1      = XY(:,3) - XY(:,2);
    T2      = XY(:,2) - XY(:,1);
    L1      = norm(T1);
    L2      = norm(T2);
    T1      = T1/L1;
    T2      = T2/L2;
    T       = [T1 T2];
    gLim    = [L1 L2];
    Xs      = -gLim(1)/2:gLim(1)/(N-1):gLim(1)/2;
    Ys      = -gLim(2)/2:gLim(2)/(N-1):gLim(2)/2;
    XY      = combvec(Xs,Ys);
    rotXY   = T*XY;
    G       = rotXY + meanG;

end

% Julien Diener (2024). 
% 2D Minimal Bounding Box 
% (https://www.mathworks.com/matlabcentral/fileexchange/31126-2d-minimal-bounding-box)
function bb = minBoundingBox(X)
    
    k  = convhull(X(1,:),X(2,:));
    CH = X(:,k);
    E  = diff(CH,1,2);           % CH edges
    T  = atan2(E(2,:),E(1,:));   % angle of CH edges (used for rotation)
    T  = unique(mod(T,pi/2));    % reduced to the unique set of first quadrant angles
    % create rotation matrix which contains
    % the 2x2 rotation matrices for *all* angles in T
    % R is a 2n*2 matrix
    R  = cos(reshape(repmat(T,2,2),2*length(T),2) ...  % duplicate angles in T
           + repmat([0 -pi ; pi 0]/2,length(T),1));   % shift angle to convert sine in cosine
    % rotate CH by all angles
    RCH = R*CH;
    % compute border size  [w1;h1;w2;h2;....;wn;hn]
    % and area of bounding box for all possible edges
    bsize = max(RCH,[],2) - min(RCH,[],2);
    area  = prod(reshape(bsize,2,length(bsize)/2));
    % find minimal area, thus the index of the angle in T 
    [~,i] = min(area);
    % compute the bound (min and max) on the rotated frame
    Rf    = R(2*i+[-1 0],:);   % rotated frame
    bound = Rf * CH;           % project CH on the rotated frame
    bmin  = min(bound,[],2);
    bmax  = max(bound,[],2);
    % compute the corner of the bounding box
    Rf      = Rf';
    bb(:,4) = bmax(1)*Rf(:,1) + bmin(2)*Rf(:,2);
    bb(:,1) = bmin(1)*Rf(:,1) + bmin(2)*Rf(:,2);
    bb(:,2) = bmin(1)*Rf(:,1) + bmax(2)*Rf(:,2);
    bb(:,3) = bmax(1)*Rf(:,1) + bmax(2)*Rf(:,2);

end