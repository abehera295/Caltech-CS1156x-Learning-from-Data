function g = error_logistic(xn, yn ,w)
%error_logistic Compute error for logistic regression
%   g = error_logistic(xn, yn ,w) computes the error of logistic regression.

g = double(log(  1.0 + exp( -yn' * xn * w )  ));
end
