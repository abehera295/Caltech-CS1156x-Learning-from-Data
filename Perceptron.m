function [w, num_iter] = Perceptron(X, y, w_init, max_iter)
% Perceptron returns the coefficient vector w and number of iterations num_iter
% for the Perceptron Learning Algorithm.  Parameters w_init and max_iter are
% optional arguments

% Get size of X and Y
[rx cx]=size(X);
ry = size(y);
if rx ~= ry
    error('size mismatch between X matrix and y vector');
end

% Fill in unset optional values.
switch nargin
    case 2
        w_init = zeros(cx,1);
        max_iter = 100000;
    case 3
        max_iter = 100000;
end

num_iter = 0;
w = w_init;

% Determine which points are misclassified initially
h = sign(X*w);
xMisClass = find(y ~= h);
numMisClass = numel(xMisClass);

while (numMisClass ~= 0 & num_iter <= max_iter)
        % Select an initial random point and calculate the initial w vector
        t = xMisClass(randi(numel(xMisClass)));
        w = w + y(t,1)*X(t,:)';

        % Determine which points are misclassified
        h = sign(X*w);
        xMisClass = find(y ~= h);

        % Increment the number of iterations and count the number of
        % misclassifieds
        num_iter = num_iter + 1;
        numMisClass = numel(xMisClass);

end;

end