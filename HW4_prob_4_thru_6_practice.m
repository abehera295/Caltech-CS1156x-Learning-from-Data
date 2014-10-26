% HW 4 prob 4, 5, 6
% Initialization
clear ; close all; clc
u = -1; v = 1;  % range of sampling interval [u, v]

h = @(x,a,b) (a.*x+b);
f = @(x) (sin(pi.*x));
z = @(x,a,b) (0.5.*(h(x,a,b) - f(x).^2));

% Calculate the average g
M = 100; % Number of runs
k = 10000; % number of data sets per run
bias_total = 0;



for j = 1:M
    g_total = zeros(1,2);
    for i = 1:k
        % Select two points at random from [u, v]^2 uniformly
        X = u + (v-u)*rand(2,2);
        % Calculate the slope and y-intercept
        a = (X(2,2)-X(2,1)) / (X(1,2)-X(1,1));
        b = X(2,1) - a.*X(1,1);

        % Calculate the total g for averaging
        g_total = g_total + [a, b]; 
    end;

    g_bar = g_total/k;

    % Calculate the bias
    bias_x = @(x) .5.*(h(x,g_bar(1),g_bar(2)) - f(x)).^2;
    bias_total = bias_total + quad(bias_x, u, v);

end;

bias = bias_total/(M*k)