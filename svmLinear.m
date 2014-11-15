function [w, b, numSV, alpha, indicesSV] = svmLinear(X, Y)
% svmLinear returns the hyperplane (w, b), number of support vectors, the
% alphas, and the indices of the support vectors for linear SVM.  The SVM is
% leared using the dual formulation of the QP-optimization problem.

% Get size of X and Y
rx=size(X, 1);
ry = size(Y);
if rx ~= ry
    error('size mismatch between X matrix and Y vector');
end

% Compute the signed X matrix
Xs = diag(Y)*X;

% Compute Q, A, f, and C  
Q = Xs*Xs';
A = [Y'; -Y'; eye(rx)];
f = -ones(rx,1);
C = zeros(size(A,1), 1);

% get alpha from QP-solver
alpha = quadprog(Q, f, -A, C);

% Determine the indices of alpha that are not 0
indicesSV = find(alpha > 0.0001);

% Number of support vectors
numSV = length(indicesSV);

% Compute w
Yn = Y(indicesSV);
Xn = X(indicesSV, :);
alpha_n = alpha(indicesSV);
w = Xn'*(alpha_n.*Yn);

% Compute b
b = Y(indicesSV(1)) - (alpha_n.*Yn)'*Xn*X(indicesSV(1),:)'; 

end