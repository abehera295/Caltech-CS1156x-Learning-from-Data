%  HW 8 problems 9, 10 

% Initialization
clear ; close all; clc
C = [0.01, 1, 100, 10^4, 10^6];
numC = length(C);

% set path to libsvm
addpath('~/Documents/Matlab/libsvm-3.20/matlab/');

% read in the train and test data
trainData = double(csvread('features.train.txt', 0, 0));
testData = double(csvread('features.test.txt', 0, 0));
trainlabels = trainData(:,1);
trainfeatures = trainData(:, 2:end);
testlabels = testData(:,1);
testfeatures = testData(:, 2:end);

%%%%%%%%%%%%%%%%%1 vs. 5 classifier
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

Ein = zeros(numC, 1);
Eout = zeros(numC, 1);

% Set options for RBF kernel
for i = 1:numC
    svmOpts = sprintf('-t 2 -g 1 -c %f -q',C(i));
    model = svmtrain(trainlabels15, trainfeatures15, svmOpts);
    [predictedLabels, accuracy, ~] = svmpredict(trainlabels15, trainfeatures15, model, '-q');
    Ein(i,1) = accuracy(2);
    [predictedLabels, accuracy, ~] = svmpredict(testlabels15, testfeatures15, model, '-q');
    Eout(i,1) = accuracy(2);
end;

[~, minEinIndex] = min(Ein);
C_lowest_Ein = C(minEinIndex)
[~, minEoutIndex] = min(Eout);
C_lowest_Eout = C(minEoutIndex)