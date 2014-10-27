function ex4_4( )
% ex4_4( )
% Finds the average hypothesis empirically
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
    
    fprintf('Average hypothesis: g(x) = %1.2fx\n', a);

end
