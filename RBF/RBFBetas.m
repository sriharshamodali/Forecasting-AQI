function betas = RBFBetas(X, centroids, memberships)


    RBFneurons = size(centroids, 1);
    sigmas = zeros(RBFneurons, 1);

    for i = 1 : RBFneurons
        
        center = centroids(i, :);
        members = X((memberships == i), :);
        differences = bsxfun(@minus, members, center);
        sqrdDiffs = sum(differences .^ 2, 2);
        distances = sqrt(sqrdDiffs);
        sigmas(i, :) = mean(distances);
    end
    betas = 1 ./ (2 .* sigmas .^ 2);
    
end