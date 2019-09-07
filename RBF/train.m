function [centers, betas, theta,J] = train(x, y, rbfneurons)

m=size(x,1);
k=rbfneurons;
iterations=100;
initial_centroids=initcentroids(x, k);
% for i=1:iterations
%     c=closestcentroids(x,initial_centroids);
%     centroids=computecentroids(x,memberships,k);
% end
[idx,centroids] = kmeans(x,k);

centers=centroids;
% betas = ones(size(centers, 1), 1) * beta;
betas = RBFBetas(x, centers, idx);
activations = zeros(m, rbfneurons);

for i = 1 : m
        input = x(i, :);
        p = RBFActivations(centers, betas, input); 
        activations(i, :) = p'; 
end

activations = bsxfun(@rdivide, activations, sum(activations, 2));

activations = [ones(m, 1), activations]; 

theta = randomlyinitializeweights(rbfneurons,1);

[J, grad]=newcostfunction(theta,activations,y);
options = optimset('MaxIter', 10000);
initial_theta = randomlyinitializeweights(rbfneurons,1);
[theta, cost]=fmincg(@(t)(newcostfunction(t, activations, y)), initial_theta, options);

end

