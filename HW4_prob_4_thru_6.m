% HW 4 prob 4, 5, 6
% Initialization
clear ; close all; clc
u = -1; v = 1;  % range of sampling interval [u, v]

h = @(x,a) (a.*x);
z = @(x,a) (0.5.*(h(x,a) - sin(pi.*x)).^2);
 
q = @(a) (quad(@(x) z(x,a), u, v));
 
% a_hat = fminunc(q, .5)
% 


% Calculate the variance
k = 1000; % number of data sets per run
M = 1000; % numer of runs
a = zeros(M, k);

a_sq_diff_total = 0; % initialize the squared difference between a and a_hat
for i=1:M
    for j = 1:k
        % Select two points at random from [u, v]^2 uniformly
        X = u + (v-u)*rand(2,2);
        % Calculate the slope of the lines through both points
        a(i,j) = (X(2,2)-X(2,1)) / (X(1,2)-X(1,1));
        % compute the squared difference between a and a_hat to contribute to
        % the expectation wrt D
    %     a_sq_diff = (a - a_hat).^2;
    %     a_sq_diff_total = a_sq_diff_total + a_sq_diff;
    end;
end;

a_hat = mean(mean(a),2)

bias_x = @(x) z(x,a_hat);
bias = quad(bias_x, u, v)

% ED = a_sq_diff_total/k
% var_x = @(x) (.5*ED.*x.^2);
% var = quad(var_x, u, v)