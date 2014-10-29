% HW 5 prob 5, 6 and 7
% Initialization
clear ; close all; clc

% Error function
E = @(w) (w(1).*exp(w(2)) - 2.*w(2).*exp(-w(1))).^2;

% Initial parameters for gradient descent
w_0 = [1; 1];
eta = 0.1;
num_iter = 0;
error_val = double( E(w_0) );
w = double(w_0);

% Loop to conduct gradient descent to minimize E
while (error_val >= 1e-14)
    grad = -HW5_prob4_7_gradient(w);
    w = w + eta.*grad;
    error_val = E(w);
    num_iter = num_iter + 1;
end;

% Output the results of gradient descent
w
num_iter
error_val

% Now run coordinate descent

% Initial parameters for coordinate descent
w_0 = [1; 1];
eta = 0.1;
num_iter = 0;
error_val = double( E(w_0) );
w = double(w_0);
N = 15; % number of iterations

for i = 1:N
    grad = -HW5_prob4_7_gradient(w);
    w(1) = w(1) + eta.*grad(1)
    grad = -HW5_prob4_7_gradient(w);
    w(2) = w(2) + eta.*grad(2)    
end;

E(w)

