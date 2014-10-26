%  Nonlinear Transformation
%  In these problems, we again apply Linear Regression for classification.
%  Consider the target function:
%  	f(x1, x2) = sign(x1^2 + x2^2 - 0.6)
%  Generate a training set of N = 1000 points on X = [-1, 1] x [-1, 1] with
%  uniform probability of picking each x in X. Generate simulated noise by
%  flipping the sign of the output in a random 10% subset of the generated
%  training set.

%  8. Carry out Linear Regression without transformation, i.e., with feature
%  vector:
%  	(1, x1, x2),
%  to find the weight w. What is the closest value to the classification
%  in-sample error Ein (run the experiment 1000 times and take the average Ein
%  in order to reduce variation in your results).
%  [a] 0
%  [b] 0.1
%  [c] 0.3
%  [d] 0.5
%  [e] 0.8

%  9. Now, transform the N = 1000 training data into the following nonlinear
%  feature vector:
% 	(1, x1, x2, x1x2, x1^2, x2^2)
%  Find the vector w_tilde that corresponds to the solution of Linear
%  Regression. Which of the following hypotheses is closest to the one you find?
%  Closest here means agrees the most with your hypothesis (has the most
%  probability of agreeing on a randomly selected point). Average over a few
%  runs to make sure your answer is stable.
%  [a] g(x1, x2) = sign(-1 - 0.05x1 + 0.08x2 + 0.13x1x2 + 1.5x1^2 + 1.5x2^2)
%  [b] g(x1, x2) = sign(-1 - 0.05x1 + 0.08x2 + 0.13x1x2 + 1.5x1^2 + 15x2^2)
%  [c] g(x1, x2) = sign(-1 - 0.05x1 + 0.08x2 + 0.13x1x2 + 15x1^2 + 1.5x2^2)
%  [d] g(x1, x2) = sign(-1 - 1.5x1 + 0.08x2 + 0.13x1x2 + 0.05x1^2 + 0.05x2^2)
%  [e] g(x1, x2) = sign(-1 - 0.05x1 + 0.08x2 + 1.5x1x2 + 0.15x1^2 + 0.15x2^2)

%  10. What is the closest value to the classification out-of-sample error Eout
%  of your hypothesis from Problem 9? (Estimate it by generating a new set of
%  1000 points and adding noise as before. Average over 1000 runs to reduce the
%  variation in your results).
%  [a] 0
%  [b] 0.1
%  [c] 0.3
%  [d] 0.5
%  [e] 0.8

% Initialization
clear ; close all; clc
InMisClassTotal = 0;
OutMisClassTotal = 0;
d = 2; % dimension of x vector
a = -1; b = 1;  % sample from [a,b]^d
N = 1000; % number of points to sample
m = 1000; % number of runs
K = 1000;  % number of points in test sample for probability calculation
g = zeros(m, d+1); % Initialize the g matrix
y = zeros(N,1); % Initialize the y vector

% Nonlinear target function
f = @(u, v) (sign(u.^2 + v.^2 - 0.6));

% Problem 8
for i=1:m
    X = a + (b-a).*rand(N, 2);   % X sample points
    X = [ones(N,1) X]; % now x = {1} X [-1,1]^d
    
    y = f(X(:,2), X(:,3)); % Generate true values
    randomIndices = randi([1 N], round(N/10, 0), 1);
    y(randomIndices,1) = -y(randomIndices,1); % Generate randomness

    % Determine w by linear regression
    w = (X'*X)\X'*y;
        
    % Determine which points are misclassified initially
    h = sign(X*w); % Convert linear regression to classification by sign
    xMisClass = find(y ~= h);
    numMisClass = numel(xMisClass);
    InMisClassTotal = InMisClassTotal + numMisClass;
end;
Ein = InMisClassTotal/(m*N) % In smaple error of linear regression

% Problem 9, 10: Now repeat using a nonlinear feature vector
w_tilde_tot = zeros(6,1);
for i=1:m
    X = a + (b-a).*rand(N, 2);   % X sample points
    % nonlinear feature vector
    X = [ones(N,1) X, X(:,1).*X(:,2), X(:,1).^2, X(:,2).^2];
    
    y = f(X(:,2), X(:,3)); % Generate true values
    randomIndices = randi([1 N], round(N/10, 0), 1);
    y(randomIndices,1) = -y(randomIndices,1); % Generate randomness

    % Determine w_tilde by linear regression
    w_tilde = (X'*X)\X'*y;       
    w_tilde_tot = w_tilde_tot + w_tilde;
    
    % Generate test point sample to compute Eout
    xOut = a + (b-a).*rand(K,2);   % X out of sample points
    xOut = [ones(K,1), xOut]; % now x = {1} X [-1,1]^d
    xOutNonLinear = [xOut, xOut(:,2).*xOut(:,3), xOut(:,2).^2, xOut(:,3).^2];
    
    % Classify xOut points using f and g
    fOut = f(xOut(:,2), xOut(:,3)); % Generate true values
    randomIndOut = randi([1 K], round(K/10, 0), 1);
    fOut(randomIndOut,1) = -fOut(randomIndOut,1); % Generate randomness  
    gOut = sign(xOutNonLinear*w_tilde);
    OutMisClassified = sum(fOut ~= gOut);
    OutMisClassTotal = OutMisClassTotal + OutMisClassified;
end;
w_tilde_avg = w_tilde_tot/m
Eout = OutMisClassTotal/(m*K) % Out of sample error of linear regression