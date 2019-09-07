function [J , grad] = costfunction(params, input_layer_size, hidden_layer_size,context_layer_size, output_layer_size, x, y, lambda)

wih = reshape(params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
[m1,n1]=size(wih);
wch = reshape(params((1+(m1*n1)):((context_layer_size*hidden_layer_size)+ (m1*n1))), hidden_layer_size, context_layer_size);
who = reshape(params((1+(context_layer_size*hidden_layer_size)+ (m1*n1)):(1+(hidden_layer_size*output_layer_size)+(context_layer_size*hidden_layer_size)+ (m1*n1))), output_layer_size, (hidden_layer_size+1));
m = size(x, 1); 
wih_grad = zeros(size(wih));
wch_grad = zeros(size(wch));
who_grad = zeros(size(who));
context=zeros(m,context_layer_size);
context(1,:)=(ones(context_layer_size,1)*0.5)';



a1=[ones(m,1) x];
s1=a1*wih';

for a=1:m
    s2(a,:)=context(a,:)*wch';
    hidden(a,:)=sigmoid(s1(a,:)+s2(a,:));
    context(a,:)=hidden(a,:);
end

hidden=[ones(m,1) hidden];
htheta=sigmoid(hidden*who');


J=(1/(2*m))*sum((htheta-y).^2);
 
regularization = lambda / (2 * m) * (sum(sum(wih(:, 2:end) .^ 2)) + sum(sum(wch(:, 2:end) .^ 2)) + sum(sum(who(:, 2:end) .^ 2)));
J = J + regularization;

for t=1:m
    yd = y(t);
    delta_3 = htheta(t,1) - yd;
    delta_2_1 = who' * delta_3 .*sigmoidgradient([1, s1(t, :)])';
    delta_2_1=delta_2_1(2:end);
    delta_2_2=who' * delta_3.*sigmoidgradient([1,s2(t,:)])';
    delta_2_2=delta_2_2(2:end);
    wih_grad= wih_grad+delta_2_1*a1(t,:);
    wch_grad=wch_grad+delta_2_2*context(t,:);
    who_grad=who_grad+delta_3*hidden(t,:);
end    

wih_grad = wih_grad / m;
wch_grad = wch_grad / m;
who_grad = who_grad / m;

wih_grad(:, 2:end) = wih_grad(:, 2:end) + lambda / m * wih(:, 2:end);
wch_grad(:, 2:end) = wch_grad(:, 2:end) + lambda / m * wch(:, 2:end);
who_grad(:, 2:end) = who_grad(:, 2:end) + lambda / m * who(:, 2:end);


grad = [wih_grad(:) ; wch_grad(:) ; who_grad(:)];
end

