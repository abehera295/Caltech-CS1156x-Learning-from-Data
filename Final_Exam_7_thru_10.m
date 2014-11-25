%  Final Exam problems 7 thru 10 

% Initialization
clear ; close all; clc

% read in the train and test data
trainData = double(csvread('features.train.txt', 0, 0));
testData = double(csvread('features.test.txt', 0, 0));
trainlabels = trainData(:,1);
trainfeatures = trainData(:, 2:end);
testlabels = testData(:,1);
testfeatures = testData(:, 2:end);
Ntrain = length(trainlabels);
Ntest = length(testlabels);

% Create training and testing labels for each digit vs. all
trainlabelsMatrix = zeros(length(trainlabels),10);
testlabelsMatrix = zeros(length(testlabels),10);

for i=0:9
    trainlabelsMatrix(:,i+1) = trainlabels;
    trainlabelsMatrix(trainlabels==i, i+1) = 1;
    trainlabelsMatrix(trainlabels~=i, i+1) = -1;  
    testlabelsMatrix(:,i+1) = testlabels;
    testlabelsMatrix(testlabels==i, i+1) = 1;
    testlabelsMatrix(testlabels~=i, i+1) = -1; 
end;

% Create index sequences for problems 7 and 8
prob7seq = [5, 6, 7, 8, 9];
prob8seq = [0, 1, 2, 3, 4];
l7 = length(prob7seq);
l8 = length(prob8seq);

% Create X vector
Xtrain = [ones(Ntrain,1), trainfeatures];
Xtest = [ones(Ntest,1), testfeatures];

% Problem 7
Z = Xtrain;
lambda = 1;
ErrorRegInSample = zeros(l7, 1);

for i=1:l7
    yTrain = trainlabelsMatrix(:, prob7seq(i)+1);
    wReg = (  Z'*Z+lambda*eye( size(Z,2) )  )\Z'*yTrain;
    yRegTrain = Z*wReg;
    hInSample = sign(yRegTrain);
    TrainMisClass = find(yTrain ~= hInSample);
    numTrainMisClass = numel(TrainMisClass);
    ErrorRegInSample(i) = numTrainMisClass/length(yTrain);
end;
ErrorRegInSample

% Problem 8
% define nonlinear transform
phi = @(x1, x2) [ones(length(x1),1), x1, x2, x1.*x2, x1.^2, x2.^2];

% Create Z for training and test sets
zTrain = phi(Xtrain(:,2), Xtrain(:,3));
zTest = phi(Xtest(:,2), Xtest(:,3));
lambda = 1;
ErrorRegOutSample = zeros(l8, 1);

for i=1:l8
    yTrain = trainlabelsMatrix(:, prob8seq(i)+1);
    yTest = testlabelsMatrix(:, prob8seq(i)+1);
    wReg = (  zTrain'*zTrain+lambda*eye( size(zTrain,2) )  )\zTrain'*yTrain;
    yRegTest = zTest*wReg;
    hOutSample = sign(yRegTest);
    TestMisClass = find(yTest ~= hOutSample);
    numTestMisClass = numel(TestMisClass);
    ErrorRegOutSample(i) = numTestMisClass/length(yTest);
end;
ErrorRegOutSample

% Problem 9
prob9seq = [prob8seq, prob7seq];
l9 = length(prob9seq);
lambda = 1;

Errors = zeros(l9, 2);

for i=1:l9
    yTrain = trainlabelsMatrix(:, prob9seq(i)+1);
    yTest = testlabelsMatrix(:, prob9seq(i)+1);
    wNot = (  Xtrain'*Xtrain+lambda*eye( size(Xtrain,2) )  )\Xtrain'*yTrain;
    wTrans = (  zTrain'*zTrain+lambda*eye( size(zTrain,2) )  )\zTrain'*yTrain;
    yNotTest = Xtest*wNot;
    yTransTest = zTest*wTrans;
    hNotOutSample = sign(yNotTest);
    hTransOutSample = sign(yTransTest);   
    NotMisClass = find(yTest ~= hNotOutSample);
    TransMisClass = find(yTest ~= hTransOutSample);
    numNotMisClass = numel(NotMisClass);
    numTransMisClass = numel(TransMisClass); 
    Error(i,1) = numNotMisClass/length(yTest);
    Error(i,2) = numTransMisClass/length(yTest);
end;

Error

%%%%%%%%%%%%%%%%% problem 10 - 1 vs. 5 classifier
% Subset the data to include data with 1 or 5 only
indicesTrain15 = find(trainlabels == 1 | trainlabels == 5);
trainData15 = trainData(indicesTrain15, :);
indicesTest15 = find(testlabels == 1 | testlabels == 5);
testData15 = testData(indicesTest15, :);

trainlabels15 = trainData15(:,1);
trainfeatures15 = trainData15(:, 2:end);
testlabels15 = testData15(:,1);
testfeatures15 = testData15(:, 2:end);

testlabels15(testlabels15 == 1) = 1;
testlabels15(testlabels15 == 5) = -1;
trainlabels15(trainlabels15 == 1) = 1;
trainlabels15(trainlabels15 == 5) = -1;

Xtrain = trainfeatures15;
Xtest = testfeatures15;
YTrain = trainlabels15;
YTest = testlabels15;
ZTrain = phi(Xtrain(:,1), Xtrain(:,2));
ZTest = phi(Xtest(:,1), Xtest(:,2));

lambda = [0.01, 1];
numLam = length(lambda);
Ein9 = zeros(numLam, 1);
Eout9 = zeros(numLam, 1);

for i = 1:numLam
    w = (  ZTrain'*ZTrain+lambda(i)*eye( size(ZTrain,2) )  )\ZTrain'*YTrain;
    yTrainLin = ZTrain*w;
    yTestLin = ZTest*w;
    hInSample = sign(yTrainLin);
    TrainMisClass = find(YTrain ~= hInSample);
    numTrainMisClass = numel(TrainMisClass);
    Ein9(i) = numTrainMisClass/length(YTrain);
    hOutSample = sign(yTestLin);
    TestMisClass = find(YTest ~= hOutSample);
    numTestMisClass = numel(TestMisClass);
    Eout9(i) = numTestMisClass /length(YTest);
end;

Ein9
Eout9
