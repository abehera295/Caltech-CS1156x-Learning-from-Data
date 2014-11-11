%  HW 7 problems 1 thru 5 

% Initialization
clear ; close all; clc

% Read in test and training sets
inSampleDat = double(csvread('in.txt', 0, 0));
outSampleDat = double(csvread('out.txt', 0, 0));

% define nonlinear transform
phi_fcn = @(x1, x2) [ones(length(x1),1), x1, x2, x1.^2, x2.^2, x1.*x2, abs(x1-x2), abs(x1+x2)];

%%%%%%%%%%%%%%%%%%%%%%% Problems 1 and 2
% Create training, test and validation sets
trainDat = inSampleDat(1:25, :);
validateDat = inSampleDat(26:end, :);
testDat = outSampleDat;

% Create vectors from the data
xTrain = trainDat(:,1:2);
yTrain = trainDat(:,3);
xVal = validateDat(:,1:2);
yVal = validateDat(:,3);
xTest = testDat(:,1:2);
yTest = testDat(:,3);

% Create the training phi matrix
phiTrain = phi_fcn(xTrain(:,1), xTrain(:,2));
phiVal = phi_fcn(xVal(:,1), xVal(:,2));
phiTest = phi_fcn(xTest(:,1), xTest(:,2));

% Apply linear regression to the list of k's
k = [3, 4, 5, 6, 7];
numSets = length(k);
ErrorLinVal1 = zeros(numSets, 1);
ErrorOutSample2 = zeros(numSets, 1);
Wreg = zeros(1,1);

for i=1:numSets
    zTrain = phiTrain(:, 1:(k(i)+1) );
    zVal = phiVal(:, 1:(k(i)+1) );
    zTest = phiTest(:, 1:(k(i)+1) );
    wLin = (zTrain'*zTrain)\zTrain'*yTrain;
    yLinVal = zVal*wLin;
    hVal = sign(yLinVal);
    ValMisClass = find(yVal ~= hVal);
    numValMisClass = numel(ValMisClass);
    ErrorLinVal1(i) = numValMisClass/length(yVal);
    yLinTest = zTest*wLin;
    hTest = sign(yLinTest);
    TestMisClass = find(yTest ~= hTest);
    numTestMisClass = numel(TestMisClass);
    ErrorOutSample2(i) = numTestMisClass/length(yTest);
end;
[M1, I1] = min(ErrorLinVal1);
minValErrorIndex1 = k(I1)

[M2, I2] = min(ErrorOutSample2);
minOutSampleErrorIndex2 = k(I2)

ErrorOutSample1 = ErrorOutSample2(I1);

%%%%%%%%%%%%%%%%%%%%%%% Problems 3 and 4
% Create training, test and validation sets
validateDat = inSampleDat(1:25, :);
trainDat = inSampleDat(26:end, :);
testDat = outSampleDat;

% Create vectors from the data
xTrain = trainDat(:,1:2);
yTrain = trainDat(:,3);
xVal = validateDat(:,1:2);
yVal = validateDat(:,3);
xTest = testDat(:,1:2);
yTest = testDat(:,3);

% Create the training phi matrix
phiTrain = phi_fcn(xTrain(:,1), xTrain(:,2));
phiVal = phi_fcn(xVal(:,1), xVal(:,2));
phiTest = phi_fcn(xTest(:,1), xTest(:,2));

% Apply linear regression to the list of k's
k = [3, 4, 5, 6, 7];
numSets = length(k);
ErrorLinVal3 = zeros(numSets, 1);
ErrorOutSample4 = zeros(numSets, 1);
Wreg = zeros(1,1);

for i=1:numSets
    zTrain = phiTrain(:, 1:(k(i)+1) );
    zVal = phiVal(:, 1:(k(i)+1) );
    zTest = phiTest(:, 1:(k(i)+1) );
    wLin = (zTrain'*zTrain)\zTrain'*yTrain;
    yLinVal = zVal*wLin;
    hVal = sign(yLinVal);
    ValMisClass = find(yVal ~= hVal);
    numValMisClass = numel(ValMisClass);
    ErrorLinVal3(i) = numValMisClass/length(yVal);
    yLinTest = zTest*wLin;
    hTest = sign(yLinTest);
    TestMisClass = find(yTest ~= hTest);
    numTestMisClass = numel(TestMisClass);
    ErrorOutSample4(i) = numTestMisClass/length(yTest);
end;
[M3, I3] = min(ErrorLinVal3);
minValErrorIndex3 = k(I3)

[M4, I4] = min(ErrorOutSample4);
minOutSampleErrorIndex4 = k(I4)

ErrorOutSample3 = ErrorOutSample4(I3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%% Problem 5
ErrorOutSample13 = [ErrorOutSample1, ErrorOutSample3]

% Compute Euclidean distances

Vec = [0.0, 0.1;
        0.1, 0.2;
        0.1, 0.3;
        0.2, 0.2;
        0.2, 0.3];
    
closest5 = zeros(size(Vec,1),1);
    
for j = 1:size(Vec,1)
    closest5(j) = norm(Vec(j,:) - ErrorOutSample13(1,:));
end;
[M5, I5] = min(closest5)

