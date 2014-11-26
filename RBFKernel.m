function [w, b] = RBFKernel(X, Y, mu, gamma)
% RBFKernel returns the hyperplane (w, b), for the solution to regression using
% a Gaussian basis function.

% Get size of X and Y
[N, d]=size(X);
ry = size(Y);
if N ~= ry
    error('size mismatch between X matrix and Y vector');
end

% Check the compatibility of mu and X
muCol = size(mu,2);
if d ~= muCol
    error('number of columns of X and mu must be the same');
end

% the kernel is always Gaussian
kernel = 'gauss';

% Fill in unset optional values.
switch nargin
    case 3
        gamma = 1;
end

% Compute the gram matrix
Z = [ones(N, 1), gram(X, mu, kernel, gamma)];

% Solve the linear regression
wlin =  (Z'*Z)\Z'*Y;
b = wlin(1);
w = wlin(2:end);

end