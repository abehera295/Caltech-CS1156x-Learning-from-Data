function ex4_5( )
% ex4_5( )
% Computes the bias
%
% PARAMETERS
%
% RETURN
    
    % Target function
    f = @(x) sin(pi*x);
    
    % Average a
    a = 0;
    
    % Number of runs
    N = 10000;
    
    for i=1:N
        % Sample two points
        x = unifrnd([-1 -1], [1 1]);
        
        % Determine a
        a_hat = (f(x(1))*x(1) + f(x(2))*x(2))/(x(1)^2 + x(2)^2);
        a = a + a_hat;
    end
    a = a/N;
    
    % Compute the bias
    b = 0;
    for i=1:N
        x = unifrnd(-1, 1);
        b = b + (a*x - f(x))^2;
    end
    b = b/N;
    
    fprintf('Bias: %f\n', b);
end
