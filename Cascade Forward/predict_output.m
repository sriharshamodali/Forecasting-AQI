function p = predict_output(theta1, theta2,theta3, x)

m = size(x, 1);
h1 = sigmoid([ones(m, 1) x] * theta1');
h2 = sigmoid(([ones(m, 1) h1] * theta2')+ ([ones(m,1) x]*theta3'));
p = h2;
end
