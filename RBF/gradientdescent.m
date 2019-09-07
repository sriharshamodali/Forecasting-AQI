function [theta, J] = gradientdescent(x, y, theta)
iterations=400; alpha=0.01;
m = length(y); 
J = zeros(iterations, 1);

for i = 1:iterations

htheta = x * theta;

    for j = 1:size(theta, 1)
        theta(j) = theta(j) - alpha / m * sum((htheta - y) .* x(:,j));
    end

    J(i) = costfunction(x, y, theta);

end
figure;
plot(1:iterations, J);

end
