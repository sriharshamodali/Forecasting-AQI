function [theta, J] = gradientdescent(x, y,theta,alpha,epochs)

m = length(y); 
J = zeros(epochs, 1);

for e = 1:epochs
n = randperm(size(x,1));
input=x(n,:);
target=y(n);
for i=1:m
htheta=input(i,:)*theta;
delta=htheta-target(i);
theta=theta-(alpha)*(delta'*input(i,:))';
end
J(e) = costfunction(x, y, theta);
    
end

end
