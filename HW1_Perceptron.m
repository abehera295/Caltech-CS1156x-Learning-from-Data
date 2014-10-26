%  The Perceptron Learning Algorithm In this problem, you will create your own
%  target function f and data set D to see how the Perceptron Learning Algorithm
%  works. Take d = 2 so you can visualize the problem, and assume X = [−1, 1] ×
%  [−1, 1] with uniform probability of picking each x ∈ X.
%  
%  In each run, choose a random line in the plane as your target function f (do
%  this by taking two random, uniformly distributed points in [−1, 1] × [−1, 1]
%  and taking the line passing through them), where one side of the line maps to
%  +1 and the other maps to −1. Choose the inputs xn of the data set as random
%  points (uniformly in X ), and evaluate the target function on each xn to get
%  the corresponding output yn. Now, in each run, use the Perceptron Learning
%  Algorithm to find g. Start the PLA with the weight vector w being all zeros,
%  and at each iteration have the algorithm choose a point randomly from the set
%  of misclassified points. We are interested in two quantities: the number of
%  iterations that PLA takes to converge to g, and the disagreement between f
%  and g which is P[f(x) 6= g(x)] (the probability that f and g will disagree on
%  their classification of a random point). You can either calculate this
%  probability exactly, or approximate it by generating a sufficiently large,
%  separate set of points to estimate it.
%  
%  In order to get a reliable estimate for these two quantities, you should
%  repeat the experiment for 1000 runs (each run as specified above) and take
%  the average over these runs.
%  
%  7. Take N = 10. How many iterations does it take on average for the PLA to
%  converge for N = 10 training points? Pick the value closest to your results
%  (again, ‘closest’ means: |your answer − given option| is closest to 0).
%  [a] 1
%  [b] 15
%  [c] 300
%  [d] 5000
%  [e] 10000
%  
%  8. Which of the following is closest to P[f(x) ~= g(x)] for N = 10?
%  [a] 0.001
%  [b] 0.01
%  [c] 0.1
%  [d] 0.5
%  [e] 0.8
%  
%  9. Now, try N = 100. How many iterations does it take on average for the PLA
%  to converge for N = 100 training points? Pick the value closest to your
%  results.
%  [a] 50
%  [b] 100
%  [c] 500
%  [d] 1000
%  [e] 5000
%  
%  10. Which of the following is closest to P[f(x) ~= g(x)] for N = 100?
%  [a] 0.001
%  [b] 0.01
%  [c] 0.1
%  [d] 0.5
%  [e] 0.8

% Initialization
clear ; close all; clc
totalIter = 0;
gMisClassTotal = 0;
d = 2; % dimension of x vector
a = -1; b = 1;  % sample from [a,b]^d
N = 100; % number of points to sample
m = 1000; % number of runs
K = 10000;  % number of points in test sample for probability calculation

%rng('default'); % set up random number generator

for i=1:m
    x = a + (b-a).*rand(2,N);   % X sample points
    x = [ones(1,N); x]; % now x = {1} X [-1,1]^d
    y = zeros(1,N);

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
    y = sign(fNormVec'*x);
    y1 = y==1;
    y2 = -(y==-1);

    [w, NumIter] = Perceptron(x', y');
    
    totalIter = totalIter + NumIter;
    
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

% Plot the points and the true function f
plot(x(2,find(y==y1)),x(3,find(y==y1)),'*',...
    x(2,find(y==y2)),x(3,find(y==y2)),'o')
hold on;
slope = (fPnts(2,2)-fPnts(2,1))/(fPnts(1,2)-fPnts(1,1));
xLeft = -1; % Whatever x value you want.
yLeft = slope * (xLeft - fPnts(1,1)) + fPnts(2,1);
xRight = 1; % Whatever x value you want.
yRight = slope * (xRight - fPnts(1,1)) + fPnts(2,1);
line([xLeft, xRight], [yLeft, yRight], 'Color', 'r', 'LineWidth', 3);