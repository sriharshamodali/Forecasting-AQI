function [J, grad] = costfunction(params, input_layer_size, hidden_layer_size, output_layer_size, x, y, lambda)

theta1 = reshape(params(1:hidden_layer_size*(input_layer_size+1)), hidden_layer_size, (input_layer_size + 1));
[m1,n1]=size(theta1);
theta2 = reshape(params((1 + (m1*n1)):(hidden_layer_size + 1 + (m1*n1))),output_layer_size, (hidden_layer_size + 1));
theta3 = reshape(params((1 + (1+hidden_layer_size+(m1*n1))):(input_layer_size+(1 + (1+hidden_layer_size+(m1*n1))))),output_layer_size,(input_layer_size+1));
             

m = size(x, 1);

J = 0;
theta1_grad = zeros(size(theta1));
theta2_grad = zeros(size(theta2));
theta3_grad = zeros(size(theta3));

a1 = [ones(m, 1) x];
z2 = a1 * theta3';
z3 = a1 * theta1';
a2=sigmoid(z3);
a2 = [ones(m, 1) a2];
z4 = a2 * theta2' + z2;
htheta = sigmoid(z4);

 J=(1/(2*m))*sum((htheta-y).^2);
 
regularization = lambda / (2 * m) * (sum(sum(theta1(:, 2:end) .^ 2)) + sum(sum(theta2(:, 2:end) .^ 2)));
J = J + regularization;

for t = 1:m
    yd = y(t);
    delta_out = htheta(t,1) - yd;
    delta_hidden = theta2' * delta_out.*sigmoidgradient([1, z3(t, :)])';
    delta_hidden = delta_hidden(2:end);
    theta1_grad = theta1_grad + delta_hidden * a1(t, :);
    theta2_grad = theta2_grad + delta_out * a2(t, :);
    theta3_grad = theta3_grad + delta_out * a1(t,:);
    
end

theta1_grad = theta1_grad / m;
theta2_grad = theta2_grad / m;
theta3_grad = theta3_grad / m;

theta1_grad(:, 2:end) = theta1_grad(:, 2:end) + lambda / m * theta1(:, 2:end);
theta2_grad(:, 2:end) = theta2_grad(:, 2:end) + lambda / m * theta2(:, 2:end);
theta3_grad(:, 2:end) = theta3_grad(:, 2:end) + lambda / m * theta3(:, 2:end);

grad = [theta1_grad(:) ; theta2_grad(:) ; theta3_grad(:)];

end







