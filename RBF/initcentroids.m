function centroids = initcentroids(x, K)

centroids = zeros(K, size(x, 2));

random = randperm(size(x, 1));
centroids = x(random(1:K), :);

end

