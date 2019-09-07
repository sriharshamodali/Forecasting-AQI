function c = closestcentroids(x, centroids)
K = size(centroids, 1);

c = zeros(size(x,1), 1); %c(i) is index of the closest centroid to example x(i)

for i=1:length(x)
    distance = inf;
    for j=1:K
        dist = norm(x(i, :) - centroids(j, :));
        if (dist < distance)
            distance = dist;
            c(i) = j;
        end
    end
end

end

