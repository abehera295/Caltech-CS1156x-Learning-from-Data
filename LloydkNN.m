function [mu, S] = LloydkNN(X, K, a ,b)
% LloydkNN returns the center points and clusters for k-nearest neighbors using
% Lloyd's algorithm.  X is the data set, K is the number of clusters, and [a,
% b]^d is the domain.  The function returns mu which are the center points and S
% which are the cluster labels.

[N, d]=size(X);  % dimensionality of the data

EinDecreasing = 1; % flag if Ein is decreasing
needRestart = 1; % flag to indicate if the algorithm must restart
Ein = 1e12;

while needRestart == 1
    % Choose K random points as centers
    mu = GenSepUniformPts(K, d, a, b);
    mu = mu(:,2:end); % Get rid of ones column
    while EinDecreasing == 1
        % Compute the L2 distance between each point and each center
        dists = L2_distance(X', mu');
        % Compute the minimum distance and identify the cluster
        [~, S] = min(dists,[],2);
        % Determine if any cluster is empty.  If so, restart the algorithm.
        if length(unique(S)) < K
            needRestart = 1;
            EinDecreasing = 1;
            break
        end;
        % Generate matrix of centers that correspond to the Xs and calculate the
        % in-sample error
        mu_vec = zeros(N, d);
        E_temp = 0;
        for i=1:N
            mu_vec(i,:) = mu(S(i),:);
            E_temp = E_temp + norm(X(i,:) - mu_vec(i,:)).^2;
        end;
        % Check if E_temp < Ein.  If so, take mu to be the centroid of the
        % clusters and loop again.  Otherwise, be done.
        if E_temp < Ein
            EinDecreasing = 1;
            needRestart = 0;
            Ein = E_temp;
            for i=1:K
                mu(i,:) = mean(X(S==i,:));
            end;
        else
            EinDecreasing = 0;
            needRestart = 0;
        end;
        
    end;
end;

end