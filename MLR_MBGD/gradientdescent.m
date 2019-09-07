function [theta, J] = gradientdescent(x, y,theta,alpha,epochs)

J = zeros(epochs, 1);
m=length(y);
batchsize=20;
for i = 1:epochs
    b=randperm(m,batchsize);
    input=x(b,:);
    target=y(b);
    htheta=input*theta;
    delta=htheta-target;
    theta=theta-(alpha/batchsize)*(delta'*input)'; 
    J(i) = costfunction(x, y, theta);

end

end
