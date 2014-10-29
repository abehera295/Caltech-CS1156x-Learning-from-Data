function grad = HW5_prob4_7_gradient(vec_init)
% HW5_prob4_7_gradient returns the gradient vector grad for the function
% E(u, v) = (u*exp(v) - 2*v*exp(-u))^2.

w = double(vec_init); % use double precision

coeff = @(w) (  2.*( w(1).*exp(w(2)) - 2.*w(2).*exp(-w(1)) )  );
E_u = @(w) (  coeff(w).*( exp(w(2)) + 2.*w(2).*exp(-w(1)) )  );
E_v = @(w) (  coeff(w).*( w(1).*exp(w(2)) - 2.*exp(-w(1)) )  );

grad = double( [E_u(w); E_v(w)] );

end