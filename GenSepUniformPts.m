function [X, Y, bndry] = GenSepUniformPts(N, d, a, b)
% GenSepUniformPts creates a data set of dimension d with N points from a
% uniform distribution on [a, b]^d.  Then, the points are classified by a
% separating hyperplane returning the y vector.

X = a + (b-a).*rand(N, d);   % X sample points
X = [ones(N, 1), X]; % now x = {1} X [-1,1]^d
Y = zeros(N, 1);
    
% Loop until there is a pattern of points that is separated, i.e., not all on
% one side of the hyperplane

success = 0;
while success==0
    PlanePnts = a + (b-a).*rand(d,d);
    M = [ones(d,1), PlanePnts];
    bndry = null(M);
    Y = sign(X*bndry);
    if ( abs( sum(Y) )==N )
        success = 0;
    else
        success = 1;
    end;    
end;