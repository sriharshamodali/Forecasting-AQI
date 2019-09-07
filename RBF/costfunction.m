function J = costfunction(x,y,theta)

m = length(y);
J = 0;
output=x*theta;
error=(output-y).^2;
J=(1/(2*m))*sum(error);
end