%%%%%5 HW 5 problems 8 and 9

% Initialization
clear ; close all; clc
totalIter = 0;
gMisClassTotal = 0;
d = 2; % dimension of x vector
a = -1; b = 1;  % sample from [a,b]^d
N = 100; % number of points to sample
m = 100; % number of runs
K = 10000;  % number of points in test sample for probability calculation

%rng('default'); % set up random number generator

for i=1:m
    X = a + (b-a).*rand(N, 2);   % X sample points
    X = [ones(N, 1), X]; % now x = {1} X [-1,1]^d
    Y = zeros(N, 1);

    % choose to random points to define function f as a lin
    fPnts = a + (b-a).*rand(2,2);

    % Define direction vector of f and normal to it
    fDirVec = [fPnts(1,2)-fPnts(1,1); fPnts(2,2)-fPnts(2,1)];
    fNormVec = [fPnts(2,2)-fPnts(2,1); fPnts(1,1)-fPnts(1,2)];

    % Equation of line f is w'*x +c = 0.  Determine c:
    c = fPnts(1,2)*fPnts(2,1) - fPnts(1,1)*fPnts(2,2);

    % Augment the fNormVec with the c in the first entry
    fNormVec = [c; fNormVec];

    % Determine true value of function f (+1, -1) and put in vector y
    Y = sign(X*fNormVec);
    Y1 = Y==1;
    Y2 = -(Y==-1);

    % Generate test point sample to compute error probability
    xTest = a + (b-a).*rand(2,K);   % X test sample points
    xTest = [ones(1,K); xTest]; % now x = {1} X [-1,1]^d
    
    % Classify xTest points using f and g
    fTest = sign(fNormVec'*xTest);
    gTest = sign(w'*xTest);
    gMisClassified = sum(fTest ~= gTest);
    gMisClassTotal = gMisClassTotal + gMisClassified;
end;
avgIter = totalIter/m;
ProbMisClass = gMisClassTotal/(m*K);