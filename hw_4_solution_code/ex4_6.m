function ex4_5( )
% ex4_4( )
% Computes the variance
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
    
    % Compute the variance
    v = 0;
    N = 150;
    for i=1:N
        % Sample two points
        x = unifrnd([-1 -1], [1 1]);

        % Determine a
        a_hat = (f(x(1))*x(1) + f(x(2))*x(2))/(x(1)^2 + x(2)^2);
        t = 0;
        for j=1:N
            x = unifrnd(-1, 1);
            t = t + (a_hat*x - a*x)^2;
        end
        v = v + t/N;
    end
    v = v/N;
    
    fprintf('Variance: %f\n', v);
end
