function [b, numSV, alpha, indicesSV] = svmKernel(X, Y, kernel, param1, param2)
% svmKernel returns the hyperplane (w, b), number of support vectors, the
% alphas, and the indices of the support vectors for kernel SVM. The SVM is
% leared using the dual formulation of the QP-optimization problem.

% Get size of X and Y
rx=size(X, 1);
ry = size(Y);
if rx ~= ry
    error('size mismatch between X matrix and Y vector');
end

% Fill in unset optional values.
switch nargin
    case 2
        kernel = 'linear';
        param1 = 1;
        param2 = 1;
    case 3
        param1 = 1;
        param2 = 1;
    case 4
        param2 = 1;
end

% Compute the gram matrix
K = gram(X, X, kernel, param1, param2);

% Compute the signed X matrix
Xs = diag(Y)*X;

% Compute Q, A, f, and C  
Q = diag(Y)*K*diag(Y);
A = [Y'; -Y'; eye(rx)];
f = -ones(rx,1);
C = zeros(size(A,1), 1);

% get alpha from QP-solver
alpha = quadprog(Q, f, -A, C);

% Determine the indices of alpha that are not 0
indicesSV = find(alpha > 0.0001);
alpha = alpha(indicesSV);

% Number of support vectors
numSV = length(indicesSV);

% Compute kernel gram matrix for support vectors
KSV = K(indicesSV,indicesSV(1));

% Compute b
Yn = Y(indicesSV);
b = Y(indicesSV(1)) - (alpha.*Yn)'*KSV; 

end