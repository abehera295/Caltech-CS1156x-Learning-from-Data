function g = grad_error_logistic(xn, yn ,w)
%grad_error_logistic Compute gradient of error for logistic regression
%   g = grad_error_logistic(z) computes the gradient of the error measure for
%   logistic regression of z.

g = (  ( -yn' * xn )./( 1.0 + exp(yn' * xn * w) )  )';
end
