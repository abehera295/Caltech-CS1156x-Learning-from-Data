%%%%%%%%%%%%% HW 4, Problems 1
% Initialization
clear ; close all; clc

%%%% #1

eps1 = 0.05;
delta = 0.05;
d_vc = 10;
f1 = @(N) (N - 8/eps1^2*log(4*(2*N)^d_vc/delta));
N1_0 = 400000; % intial guess

N1 = fzero(f1,N1_0)