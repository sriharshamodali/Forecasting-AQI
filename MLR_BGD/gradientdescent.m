function [theta, J] = gradientdescent(x, y,theta,alpha,epochs)

m = length(y); 
J = zeros(epochs, 1);

for i = 1:epochs

htheta = x * theta;
delta=htheta-y;
theta=theta-(alpha/m)*(delta'*x)'; 
J(i) = costfunction(x, y, theta);

end

end
