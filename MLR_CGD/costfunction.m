function [J, grad]=costfunction(theta,x,y)

m=length(y);

J=0;
grad=zeros(size(theta));
output=x*theta;
error=(output-y).^2;
J=(1/(2*m))*sum(error);

for i=1:size(theta,1)
    grad(i)=(1/m)*sum((output-y).*x(:,i));
end


end