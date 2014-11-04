%  HW 6 problems 2 thru 6 

% Initialization
clear ; close all; clc

% Read in test and training sets
testDat = double(csvread('in.txt', 0, 0));
trainDat = double(csvread('out.txt', 0, 0));

% define nonlinear transform
phi = @(x1, x2) [ones(length(x1),1), x1, x2, x1.^2, x2.^2, x1.*x2, abs(x1-x2), abs(x1+x2)];

% Create vectors from the data
yTest = testDat(:,3);
yTrain = trainDat(:,3);
xTest = testDat(:,1:2);
xTrain = trainDat(:,1:2);

% Create Z for training and test sets
zTrain = phi(xTrain(:,1), xTrain(:,2));
zTest = phi(xTest(:,1), xTest(:,2));

% Linear regression on Z
wLin = (zTrain'*zTrain)\zTrain'*yTrain;

% Apply w_lin to training and test data
yLinTrain = zTrain*wLin;
yLinTest = zTest*wLin;

% Compute the in-sample and out-of-sample errors
hInSample = sign(yLinTrain);
TrainMisClass = find(yTrain ~= hInSample);
numTrainMisClass = numel(TrainMisClass);
ErrorLinInSample = numTrainMisClass/length(yTrain)
hOutSample = sign(yLinTest);
TestMisClass = find(yTest ~= hOutSample);
numTestMisClass = numel(TestMisClass);
ErrorLinOutSample = numTestMisClass/length(yTest)

% Compute Euclidean distances
ErrorsLin = [ErrorLinInSample, ErrorLinOutSample];
Vec2 = [0.03, 0.08;
        0.03, 0.10;
        0.04, 0.09;
        0.04, 0.11;
        0.05, 0.10];
    
closest = zeros(size(Vec2,1),1);
    
for i = 1:size(Vec2,1)
    closest(i) = norm(Vec2(i,:) - ErrorsLin);
end;
closest

% Add weight decay
