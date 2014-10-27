function ex4_7( )
% ex4_7( )
% Compares the expected out-of-sample error of several hypothesis sets. 
%
% PARAMETERS
%
% RETURN
    
    % Target function
    f = @(x) sin(pi*x);
    
    % Average a
    a = 0;
    
    % Number of runs
    N = 20000;
    
    % All of the given hypothesis are polynomials of degree 2
    % We provide different learning algorithms for each type
    
    fprintf('expected E_out A: %f\n', expectedOutOfSampleError(@learnA, f, N));
    fprintf('expected E_out B: %f\n', expectedOutOfSampleError(@learnB, f, N));
    fprintf('expected E_out C: %f\n', expectedOutOfSampleError(@learnC, f, N));
    fprintf('expected E_out D: %f\n', expectedOutOfSampleError(@learnD, f, N));
    fprintf('expected E_out E: %f\n', expectedOutOfSampleError(@learnE, f, N));
end

function e = expectedOutOfSampleError(learn, f, N)
% Approximates the expected out-of-sample error
%
% PARAMETERS
% learn - [function] The training function
% f     - [function] The target function
% N     - [1x1] Number of runs
%
% RETURN
% e     - [1x1] Expected out-of-sample error

    a = [0 0 0];
    
    % Determine the average hypothesis
    for i=1:N
        % Sample two points
        x = unifrnd([-1 -1], [1 1]);
        
        % Determine a
        a_hat = learn(x, f);
        
        a = a + a_hat;
    end
    a = a/N;
    
    % Compute the variance 
    v = 0;
    for i=1:floor(sqrt(N))
        % Sample two points
        x = unifrnd([-1 -1], [1 1]);

        % Determine a
        w = learn(x, f);
        t = 0;
        for j=1:floor(sqrt(N))
            x = unifrnd(-1, 1);
            bx = [x^2; x; 1];
            t = t + (a*bx - w*bx)^2;
        end
        v = v + t/floor(sqrt(N));
    end
    v = v/floor(sqrt(N));
    
    % Compute the squared bias
    % Compute the bias
    b = 0;
    for i=1:N
        x = unifrnd(-1, 1);
        bx = [x^2; x; 1];
        b = b + (a*bx - f(x))^2;
    end
    b = b/N;
    
    e = b+v;
end

function w = learnA(x, f)
% Trains the parameters for the hypothesis h(x) = c
%
% PARAMETERS
% x - [2x1] The training points
% f - [function] The target function
% 
% RETURN
% w - [1x3] The weight vector
    A = [1;1];
    b = [f(x(1)); f(x(2))];
    w_hat = A\b;
    w = [0, 0, w_hat(1)];
end

function w = learnB(x, f)
% Trains the parameters for the hypothesis h(x) = bx
%
% PARAMETERS
% x - [2x1] The training points
% f - [function] The target function
% 
% RETURN
% w - [1x3] The weight vector

    A = [x(1);x(2)];
    b = [f(x(1)); f(x(2))];
    w_hat = A\b;
    w = [0, w_hat(1), 0];
end

function w = learnC(x, f)
% Trains the parameters for the hypothesis h(x) = bx + c
%
% PARAMETERS
% x - [2x1] The training points
% f - [function] The target function
% 
% RETURN
% w - [1x3] The weight vector
    A = [x(1), 1;x(2), 1];
    b = [f(x(1)); f(x(2))];
    w_hat = A\b;
    w = [0, w_hat(1), w_hat(2)];
end

function w = learnD(x, f)
% Trains the parameters for the hypothesis h(x) = ax^2
%
% PARAMETERS
% x - [2x1] The training points
% f - [function] The target function
% 
% RETURN
% w - [1x3] The weight vector
    A = [x(1)^2;x(2)^2];
    b = [f(x(1)); f(x(2))];
    w_hat = A\b;
    w = [w_hat(1), 0, 0];
end

function w = learnE(x, f)
% Trains the parameters for the hypothesis h(x) = ax^2 + b
%
% PARAMETERS
% x - [2x1] The training points
% f - [function] The target function
% 
% RETURN
% w - [1x3] The weight vector
    A = [x(1)^2, 1;x(2)^2, 1];
    b = [f(x(1)); f(x(2))];
    w_hat = A\b;
    w = [w_hat(1), 0, w_hat(2)];
end
