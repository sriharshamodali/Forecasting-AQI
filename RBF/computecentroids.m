function centroids = computecentroids(x, c, K)
[m, n] = size(x);
centroids = zeros(K, n);
for i=1:K
    indices = c == i;
    for j=1:n
        centroids(i, j) = sum(x(:, j) .* indices) / sum(indices);
    end
end
end

