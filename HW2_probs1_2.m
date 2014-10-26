% Hoeffding Inequality
% Run a computer simulation for flipping 1,000 virtual fair coins. Flip each
% coin independently 10 times. Focus on 3 coins as follows: c1 is the first
% coin flipped, crand is a coin chosen randomly from the 1,000, and cmin is the
% coin which had the minimum frequency of heads (pick the earlier one in case of
% a tie). Let nu_1, nu_rand, and nu_min be the fraction of heads obtained for
% the 3 respective coins out of the 10 tosses.

% Run the experiment 100,000 times in order to get a full distribution of nu_1,
% nu_rand, and nu_min (note that crand and cmin will change from run to run).

%1. The average value of nu_min is closest to:
% [a] 0
% [b] 0.01
% [c] 0.1
% [d] 0.5
% [e] 0.67

% 2. Which coin(s) has a distribution of nu that satisfies Hoeffding's
% Inequality?
% [a] c1 only
% [b] crand only
% [c] cmin only
% [d] c1 and crand
% [e] cmin and crand

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization
clear ; close all; clc
M = 100000; % number of simulations
nu_min_total = 0;
nu_1_total = 0;
nu_rand_total = 0;

for i=1:M
    % Flip 1,000 fair coins 10 times each
    P = 0.5; % probability of success, fair coin
    Ncoins = 1000; % number of coins
    k = 10; % number of flips per coin

    % number of heads for each set of flips
    flip_results = binornd(k, P, Ncoins, 1); 
    nu = flip_results/k; % fraction of heads
    nu_rand = nu(randi([1,1000])); % choose random nu
    nu_min = min(nu); % find minimum of nu
    nu_min_total = nu_min_total + nu_min;
    nu_1_total = nu_1_total + nu(1);
    nu_rand_total = nu_rand_total + nu_rand;
end;

nu_min_avg = nu_min_total/M;
nu_1_avg = nu_1_total/M;
nu_rand_avg = nu_rand_total/M;