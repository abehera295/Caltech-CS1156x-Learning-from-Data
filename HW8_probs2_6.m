%  HW 8 problems 2 thru 6 

% Initialization
clear ; close all; clc

% set data directory
dirData = './';

% read in the train and test data
trainData = double(csvread('features.train.txt', 0, 0));
testData = double(csvread('features.test.txt', 0, 0));
trainlabels = trainData(:,1);
trainfeatures = trainData(:, 2:end);
testlabels = testData(:,1);
testfeatures = testData(:, 2:end);
%[trainlabels, trainfeatures ]= libsvmread('features.train.txt');

% Create training labels for each digit vs. all
trainlabelsMatrix = zeros(length(trainlabels),10);

for i=0:9
    trainlabelsMatrix(:,i+1) = trainlabels;
    trainlabelsMatrix(trainlabels==i, i+1) = 1;
    trainlabelsMatrix(trainlabels~=i, i+1) = -1;  
end;

% Create index sequences for problems 2 and 3
prob2seq = [0, 2, 4, 6, 8];
prob3seq = [1, 3, 5, 7, 9];
l2 = length(prob2seq);
l3 = length(prob3seq);
MSE2 = zeros(l2, 1);
MSE3 = zeros(l3, 1);

% train the svm classifier
for i=1:l2;
    model = svmtrain(trainlabelsMatrix(:,prob2seq(i)+1), trainfeatures, ...
    '-t 1 -d 2 -r 1, -g 1 -c 0.01');
    [predictedLabels, accuracy, ~] = svmpredict(trainlabelsMatrix(:,prob2seq(i)+1), ...
        trainfeatures, model, '-q');
    MSE2(i,1) = accuracy(2);
end;

for i=1:l3;
    model = svmtrain(trainlabelsMatrix(:,prob3seq(i)+1), trainfeatures, ...
    '-t 1 -d 2 -r 1, -g 1 -c 0.01');
    [predictedLabels, accuracy, ~] = svmpredict(trainlabelsMatrix(:,prob3seq(i)+1), ...
        trainfeatures, model, '-q');
    MSE3(i,1) = accuracy(2);
end;

% Training error for problems 2 and 3
MSE2
MSE3

% problem 5 and 6 - 1 vs. 5 classifier


