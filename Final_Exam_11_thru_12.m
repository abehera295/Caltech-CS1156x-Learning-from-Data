%  Final Exam problems 11, 12 

% Initialization
clear ; close all; clc
Q = 2; % degree of the kernel polynomial

% set path to libsvm
addpath('~/Documents/Matlab/libsvm-3.20/matlab/');

% Create the data matrices
X = [1, 0;
    0, 1;
    0, -1;
    -1, 0;
    0, 2;
    0, -2;
    -2, 0];

Yneg = -ones(3,1);
Ypos = ones(4,1);
Y = [Yneg; Ypos];

numNegs = length(Yneg);

Xneg = X(1:numNegs,:);
Xpos = X((numNegs+1):end, :);

% Nonlinear transform
phi = @(x1, x2) [x2.^2 - 2*x1 - 1, x1.^2 - 2*x2 + 1];

% Generate Z matrices
Zpos = phi(Xpos(:,1), Xpos(:,2));
Zneg = phi(Xneg(:,1), Xneg(:,2));
Z = [Zneg; Zpos];

% Plot the data in Z space
% Plot the points and the true function f
plot(Zpos(:,1),Zpos(:,2),'+',Zneg(:,1),Zneg(:,2),'o')
axis([min(Z(:,1))-1, max(Z(:,1))+1, min(Z(:,2))-1, max(Z(:,2))+1])

% Run SVm with polynomial kernel
[bSVM, SupportVecs, alpha, indicesSV] = svmKernel(X,Y,'poly',1,2);
SupportVecs
