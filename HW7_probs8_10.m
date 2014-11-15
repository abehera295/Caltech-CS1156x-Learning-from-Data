%  HW 7 problems 8 thru 10 

% Initialization
clear ; close all; clc
SVMBetterPLA = 0;
TotalSupportVecs = 0;
d = 2; % dimension of x vector
a = -1; b = 1;  % sample from [a,b]^d
N = 10; % number of points to sample
m = 1000; % number of runs
K = 5000;  % number of points in test sample for probability calculation

for i=1:m
    %Get train dataset
    [X, Y, fNormVec] = GenSepUniformPts(N, d, a, b);
    
    % Generate test sample to compute out of sample error
    xTest = [ones(K, 1), a + (b-a).*rand(K, 2)]; % now x = {1} X [-1,1]^d
    
    % Classify xTest points using f and g
    ytest = sign(xTest*fNormVec);
    
    % Classify using PLA
    w_init = zeros(d+1, 1);
    [wPLA, NumIter] = Perceptron(X, Y, w_init);
     
    % Compute out-of-sample error for PLA
    gPLA = sign(xTest*wPLA);
    ProbMisClassPLA = sum(ytest ~= gPLA)/K;
    
    % Run SVM
    XSVM = X(:,2:end);
    [wSVM, bSVM, SupportVecs, alpha, indicesSV] = svmLinear(XSVM,Y);
    
    % Compute out-of-sample error for SVM
    xSVMtest = xTest(:,2:end);
    gSVM = sign(xSVMtest*wSVM + bSVM);
    ProbMisClassSVM = sum(ytest ~= gSVM)/K;
    
    % Tally the number of times SVM has a better classification probability
    SVMBetterPLA = SVMBetterPLA + (ProbMisClassSVM < ProbMisClassPLA);
    
    % Tally the average number of support vectors
    TotalSupportVecs = TotalSupportVecs + SupportVecs;
end;
PercentSVMBetterPLA = SVMBetterPLA/m*100;
fprintf('SVM is better than PLA %.2f%% of the time\n', PercentSVMBetterPLA);
AvgNumSupportVecs = TotalSupportVecs/m;
fprintf('The average number of support vectors is %.2f\n', AvgNumSupportVecs);