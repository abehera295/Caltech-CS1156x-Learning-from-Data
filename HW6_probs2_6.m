%  HW 6 problems 2 thru 6 

% Initialization
clear ; close all; clc

% Read in test and training sets
trainDat = double(csvread('in.txt', 0, 0));
testDat = double(csvread('out.txt', 0, 0));

% define nonlinear transform
phi = @(x1, x2) [ones(length(x1),1), x1, x2, x1.^2, x2.^2, x1.*x2, abs(x1-x2), abs(x1+x2)];

% Create vectors from the data
yTest = testDat(:,3);
yTrain = trainDat(:,3);
xTest = testDat(:,1:2);
xTrain = trainDat(:,1:2);

% Lambda vector for problems 2, 3, 4, 5, 6
lambda = [0, 10^(-3), 10^3, 10^2, 10, 1, 10^(-1), 10^(-2), 10^(-4)];
numLambdas = length(lambda);

% Create Z for training and test sets
zTrain = phi(xTrain(:,1), xTrain(:,2));
zTest = phi(xTest(:,1), xTest(:,2));

ErrorRegInSample = zeros(numLambdas,1);
ErrorRegOutSample = zeros(numLambdas,1);
ErrorsReg = zeros(numLambdas, 2);

for i=1:numLambdas
    % Linear regression on Z
    wReg = (  zTrain'*zTrain+lambda(i)*eye( size(zTrain,2) )  )\zTrain'*yTrain;

    % Apply w_lin to training and test data
    yRegTrain = zTrain*wReg;
    yRegTest = zTest*wReg;

    % Compute the in-sample and out-of-sample errors
    hInSample = sign(yRegTrain);
    TrainMisClass = find(yTrain ~= hInSample);
    numTrainMisClass = numel(TrainMisClass);
    ErrorRegInSample(i) = numTrainMisClass/length(yTrain);
    hOutSample = sign(yRegTest);
    TestMisClass = find(yTest ~= hOutSample);
    numTestMisClass = numel(TestMisClass);
    ErrorRegOutSample(i) = numTestMisClass/length(yTest);
    ErrorsReg(i,:) = [ErrorRegInSample(i), ErrorRegOutSample(i)];
end;

ErrorsReg

% Compute Euclidean distances problem 2

Vec2 = [0.03, 0.08;
        0.03, 0.10;
        0.04, 0.09;
        0.04, 0.11;
        0.05, 0.10];
    
closest2 = zeros(size(Vec2,1),1);
    
for j = 1:size(Vec2,1)
    closest2(j) = norm(Vec2(j,:) - ErrorsReg(1,:));
end;
closest2

% Compute Euclidean distances problem 3
Vec3 = [0.01, 0.02;
        0.02, 0.04;
        0.02, 0.06;
        0.03, 0.08;
        0.03, 0.10];
    
closest3 = zeros(size(Vec3,1),1);
    
for j = 1:size(Vec3,1)
    closest3(j) = norm(Vec3(j,:) - ErrorsReg(2,:));
end;
closest3

% Compute Euclidean distances problem 4
Vec4 = [0.2, 0.2;
        0.2, 0.3;
        0.3, 0.3;
        0.3, 0.4;
        0.4, 0.4];
    
closest4 = zeros(size(Vec4,1),1);
    
for j = 1:size(Vec4,1)
    closest4(j) = norm(Vec4(j,:) - ErrorsReg(3,:));
end;
closest4

