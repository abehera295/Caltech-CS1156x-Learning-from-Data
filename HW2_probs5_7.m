%  Linear Regression
%  In these problems, we will explore how Linear Regression for classification
%  works. As with the Perceptron Learning Algorithm in Homework # 1, you will
%  create your own target function f and data set D. Take d = 2 so you can
%  visualize the problem, and assume X = [-1; 1] x [-1; 1] with uniform
%  probability of picking each x in X. In each run, choose a random line in the
%  plane as your target function f (do this by taking two random, uniformly
%  distributed points in [-1; 1] x [-1; 1] and taking the line passing through
%  them), where one side of the line maps to +1 and the other maps to -1. Choose
%  the inputs xn of the data set as random points (uniformly in X), and evaluate
%  the target function on each xn to get the corresponding output yn.

%  5. Take N = 100. Use Linear Regression to find g and evaluate Ein, the
%  fraction of in-sample points which got classified incorrectly. Repeat the
%  experiment 1000 times and take the average (keep the g's as they will be used
%  again in Problem 6). Which of the following values is closest to the average
%  Ein? (closest is the option that makes the expression |your answer - given
%  option| closest to 0. Use this definition of closest here and throughout)
%  [a] 0
%  [b] 0.001
%  [c] 0.01
%  [d] 0.1
%  [e] 0.5

%  6. Now, generate 1000 fresh points and use them to estimate the out-of-sample
%  error Eout of g that you got in Problem 5 (number of misclassified
%  out-of-sample points / total number of out-of-sample points). Again, run the
%  experiment 1000 times and take the average. Which value is closest to the
%  average Eout?
%  [a] 0
%  [b] 0.001
%  [c] 0.01
%  [d] 0.1
%  [e] 0.5

%  7. Now, take N = 10. After finding the weights using linear regression, use
%  them as a vector of initial weights for the Perceptron Learning Algorithm.
%  Run PLA until it converges to a final vector of weights that completely
%  separates all the in-sample points. Among the choices below, what is the
%  closest value to the average number of iterations (over 1000 runs) that PLA
%  takes to converge? (When implementing PLA, have the algorithm choose a point
%  randomly from the set of misclassified points at each iteration)
%  [a] 1
%  [b] 15
%  [c] 300
%  [d] 5000
%  [e] 10000

% Initialization
clear ; close all; clc
InMisClassTotal = 0;
OutMisClassTotal = 0;
d = 2; % dimension of x vector
a = -1; b = 1;  % sample from [a,b]^d
N = 100; % number of points to sample
m = 1000; % number of runs
K = 1000;  % number of points in test sample for probability calculation
g = zeros(m, d+1); % Initialize the g matrix
y = zeros(N,1); % Initialize the y vector

%rng('default'); % set up random number generator

for i=1:m
    X = a + (b-a).*rand(N, 2);   % X sample points
    X = [ones(N,1) X]; % now x = {1} X [-1,1]^d

    % choose to random points to define function f as a lin
    fPnts = a + (b-a).*rand(2,2);

    % Define direction vector of f and normal to it
    fDirVec = [fPnts(1,2)-fPnts(1,1); fPnts(2,2)-fPnts(2,1)];
    fNormVec = [fPnts(2,2)-fPnts(2,1); fPnts(1,1)-fPnts(1,2)];

    % Equation of line f is w'*x +c = 0.  Determine c:
    c = fPnts(1,2)*fPnts(2,1) - fPnts(1,1)*fPnts(2,2);

    % Augment the fNormVec with the c in the first entry
    f_w = [c; fNormVec];

    % Determine true value of function f (+1, -1) and put in vector y
    y = sign(X*f_w);
    
    % Determine g by linear regression
    gtemp = (X'*X)\X'*y;
    g(i,:) = gtemp;
    
    % Determine which points are misclassified initially
    h = sign(X*gtemp); % Convert linear regression to classification by sign
    xMisClass = find(y ~= h);
    numMisClass = numel(xMisClass);
    InMisClassTotal = InMisClassTotal + numMisClass;
    
    % Generate test point sample to compute Eout
    xOut = a + (b-a).*rand(K,2);   % X out of sample points
    xOut = [ones(K,1), xOut]; % now x = {1} X [-1,1]^d
    
    % Classify xOut points using f and g
    fOut = sign(xOut*f_w);
    gOut = sign(xOut*gtemp);
    OutMisClassified = sum(fOut ~= gOut);
    OutMisClassTotal = OutMisClassTotal + OutMisClassified;
end;
Ein = InMisClassTotal/(m*N) % In smaple error of linear regression
Eout = OutMisClassTotal/(m*K) % Out of sample error of linear regression

% Initial values for comparison to Perceptron
N = 10; % number of points
InMisClassTotal = 0;
totalIter = 0;
y = zeros(N,1); % Initialize the y vector

for i=1:m
    X = a + (b-a).*rand(N, 2);   % X sample points
    X = [ones(N,1) X]; % now x = {1} X [-1,1]^d

    % choose to random points to define function f as a lin
    fPnts = a + (b-a).*rand(2,2);

    % Define direction vector of f and normal to it
    fDirVec = [fPnts(1,2)-fPnts(1,1); fPnts(2,2)-fPnts(2,1)];
    fNormVec = [fPnts(2,2)-fPnts(2,1); fPnts(1,1)-fPnts(1,2)];

    % Equation of line f is w'*x +c = 0.  Determine c:
    c = fPnts(1,2)*fPnts(2,1) - fPnts(1,1)*fPnts(2,2);

    % Augment the fNormVec with the c in the first entry
    f_w = [c; fNormVec];

    % Determine true value of function f (+1, -1) and put in vector y
    y = sign(X*f_w);
    
    % Determine w by linear regression
    wLinReg = (X'*X)\X'*y;
    
    % Use perceptron with x initially from linear regression
    [x, NumIter] = Perceptron(X, y, wLinReg);
    
    totalIter = totalIter + NumIter;
end;

avgPLAIter = totalIter/m