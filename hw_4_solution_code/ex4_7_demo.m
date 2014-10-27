function ex4_7_demo( )
% ex4_7( )
% Plots the different hypothesis for a random pair of points
%
% PARAMETERS
%
% RETURN
    
    % Target function
    f = @(x) sin(pi*x);
    
    figure;
    hold on;
    
    fplot(f, [-1,1], 'r');
    x = unifrnd([-1 -1], [1 1]);
    plot(x, f(x), 'x');
    
    w = learnA(x, f);
    fplot(@(x) w(1)*x^2 + w(2)*x + w(3), [-1 1], 'b');
    w = learnB(x, f);
    fplot(@(x) w(1)*x^2 + w(2)*x + w(3), [-1 1], 'g');
    w = learnC(x, f);
    fplot(@(x) w(1)*x^2 + w(2)*x + w(3), [-1 1], 'k');
    w = learnD(x, f);
    fplot(@(x) w(1)*x^2 + w(2)*x + w(3), [-1 1], 'm');
    w = learnE(x, f);
    fplot(@(x) w(1)*x^2 + w(2)*x + w(3), [-1 1], 'y');
    plot(x, f(x), 'dk');
    
    legend('target function', 'A', 'B', 'C', 'D', 'E', 'samples');
    % All of the given hypothesis are polynomials of degree 2
    % We provide different learning algorithms for each type
    
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
