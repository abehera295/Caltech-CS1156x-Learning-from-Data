%%%%%%%%%%%%% HW 4, Problems 2, 3
% Initialization
clear ; close all; clc

%%%% #2
delta = 0.05;
d_vc = 50;
M1 = 10005;
M0 = M1 - 10;
N = M0:M1;
fa = @(N) (  sqrt( 8./N .* log(4.*(2.*N).^d_vc./delta) )  );
fb = @(N) (  sqrt( 2./N.*log(2.*N.^(d_vc+1)) ) + sqrt( 2./N.*log(1./delta) ) + 1./N  );
fd1 = @(N) (  log( (4/delta)^(1/(2*N))*N^(d_vc/N) )  );
y2c = zeros(1,M1-M0+1);
y2d = zeros(1,M1-M0+1);
eps0 = 0.1; % initial condition for c and d

for k=M0:M1
    fc = @(eps) (   sqrt(1./k*(  2.*eps + log((6.*(2.*k).^d_vc)./delta)  )) - eps );
    y2c(k-M0+1) = fzero(fc,eps0);
end;

for k=M0:M1
    fd = @(eps) (  sqrt( 1/k*2*eps*(1+eps) + fd1(k) ) - eps  );
    y2d(k-M0+1) = fzero(fd, eps0);
end;

y2a = fa(N);
y2b = fb(N);
plot(N,y2a,'r',N,y2b,'b',N,y2c,'g',N,y2d,'k*', 'LineWidth', 3)
legend('VC','Rademacher','Parrondo','Devroye')