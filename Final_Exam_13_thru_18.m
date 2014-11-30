%  Final Exam problems 13 through 18 

% Initialization
clear ; close all; clc
d = 2; % dimension of x vector
a = -1; b = 1;  % sample from [a,b]^d
N = 100; % number of points to sample
m = 1000; % number of runs
T = 1000; % number of points in test sample
gamma = 1.5;
K = 9; % Number of clusters
ErrorSVMInSample = zeros(m,1);
ErrorRBFInSample = zeros(m,1);
ErrorSVMOutSample = zeros(T,1);
ErrorRBFOutSample = zeros(T,1);
SVMbeatRBF = 0;

% Target function
f = @(x) sign(  x(:,2) - x(:,1) + .25.*sin( pi*x(:,1) )  );

for i=1:m
    % Get train dataset
    X = GenSepUniformPts(N, d, a, b);
    X = X(:,2:end);
    Y = f(X);
    
    % Run hard margin SVM
    [bSVM, SupportVecs, alpha, indicesSV] = svmKernel(X,Y,'gauss',gamma);
    
    % Get the center points
    mu = LloydkNN(X, K, a ,b);
    
    % Compute the gram matrices
    KSVM = gram(X, X, 'gauss', gamma);
    KSVM_SV = KSVM(indicesSV,:);
    KRBF = gram(X, mu, 'gauss', gamma);
        
    % Compute the RBf hypothesis
    [wRBF, bRBF] = RBFKernel(X, Y, mu, gamma);
    
    % Compute the SVM and RBF hypothesis and in-sample error
    gSVM = sign(KSVM_SV'*(alpha.*Y(indicesSV)) + bSVM);
    SVMTrainMisClass = find(Y ~= gSVM);
    numSVMTrainMisClass = numel(SVMTrainMisClass);
    ErrorSVMInSample(i) = numSVMTrainMisClass/length(Y);
    gRBF = sign(KRBF*wRBF + bRBF);
    RBFTrainMisClass = find(Y ~= gRBF);
    numRBFTrainMisClass = numel(RBFTrainMisClass);
    ErrorRBFInSample(i) = numRBFTrainMisClass/length(Y);
    
    % Generate test dataset
    Xtest = GenSepUniformPts(T, d, a, b);
    Xtest = Xtest(:,2:end);
    Ytest = f(Xtest);
    
    % Compute the gram matrics for SVM and RBF for test data
    XtestSV = Xtest(indicesSV,:);
    KSVM_test = gram(Xtest, X(indicesSV,:), 'gauss', gamma);
    KRBF_test = gram(Xtest, mu, 'gauss', gamma);
    
    % Predict using SVM and RBF
    gRBF_test = sign(KRBF_test*wRBF + bRBF);
    gSVM_test = sign(KSVM_test*(alpha.*Y(indicesSV)) + bSVM);
    
    % Calculate the out-of-sample error
    RBFTestMisClass = find(Ytest ~= gRBF_test);
    numRBFTestMisClass = numel(RBFTestMisClass);
    ErrorRBFOutSample(i) = numRBFTestMisClass/length(Ytest);
    SVMTestMisClass = find(Ytest ~= gSVM_test);
    numSVMTestMisClass = numel(SVMTestMisClass);
    ErrorSVMOutSample(i) = numSVMTestMisClass/length(Ytest);
    
    % Calculate the number of times that SVM beats RBF
    SVMbeatRBF = SVMbeatRBF + (ErrorRBFOutSample(i) < ErrorSVMOutSample(i));
    
end;
notSVMseparable = find(ErrorSVMInSample ~= 0);
numNotSVMseparable = numel(notSVMseparable)

PercentSVMBeatRBF = SVMbeatRBF/m

ErrorRBFInSampleAvg = sum(ErrorRBFInSample)/m
ErrorRBFOutSampleAvg = sum(ErrorRBFOutSample)/m

RBFEinZero = find(ErrorRBFInSample == 0);
numRBFEinZero = numel(RBFEinZero);

PercentRBFEinZero = numRBFEinZero/m