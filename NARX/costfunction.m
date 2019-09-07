function [J grad] = costfunction(params, input_layer_size, hidden_layer_size, output_layer_size, x, y)

wih = reshape(params(1:hidden_layer_size*(input_layer_size+1)), hidden_layer_size, (input_layer_size + 1));
[m1,n1]=size(wih);
who = reshape(params((1 + (m1*n1)):(hidden_layer_size + 1 + (m1*n1))),output_layer_size, (hidden_layer_size + 1));
wi1h = reshape(params((2+hidden_layer_size+(m1*n1)):((input_layer_size*hidden_layer_size)+1+hidden_layer_size+(m1*n1))),hidden_layer_size,input_layer_size);
wi2h = reshape(params(((input_layer_size*hidden_layer_size)+2+hidden_layer_size+(m1*n1)):((2*input_layer_size*hidden_layer_size)+1+hidden_layer_size+(m1*n1))),hidden_layer_size,input_layer_size);
wo1h = reshape(params(((2*input_layer_size*hidden_layer_size)+2+hidden_layer_size+(m1*n1)):((2*input_layer_size*hidden_layer_size)+1+(2*hidden_layer_size)+(m1*n1))),hidden_layer_size,output_layer_size);
wo2h = reshape(params(((2*input_layer_size*hidden_layer_size)+2+(2*hidden_layer_size)+(m1*n1)):((2*input_layer_size*hidden_layer_size)+1+(3*hidden_layer_size)+(m1*n1))),hidden_layer_size,output_layer_size);
          
m = size(x, 1);

J = 0;
wih_grad = zeros(size(wih));
who_grad = zeros(size(who));
wi1h_grad = zeros(size(wi1h));
wi2h_grad = zeros(size(wi2h));
wo1h_grad = zeros(size(wo1h));
wo2h_grad = zeros(size(wo2h));
hidden=zeros(m,hidden_layer_size);
htheta=zeros(m,1);

i1h=[zeros(1,hidden_layer_size);x(2:end,:)];
i2h=[zeros(2,hidden_layer_size);x(3:end,:)];

ih = [ones(m, 1) x];   
s1 = ih * wih';
s2 = i1h * wi1h';
s3 = i2h * wi2h';
s4 = zeros(m,hidden_layer_size);
s5 = zeros(m,hidden_layer_size);
for k=1:m
    hidden(k,:)=sigmoid(s1(k,:)+s2(k,:)+s3(k,:)+s4(k,:)+ s5(k,:));
    s6(k,:)=[1 hidden(k,:)]*who';
    htheta(k)=sigmoid(s6(k,:));
    s4(k+1,:)=htheta(k,:)*wo1h';
    s5(k+2,:)=htheta(k,:)*wo2h';
end
hidden=[ones(m,1) hidden];
o1h=[zeros(1,1);htheta(2:end,:)]; o2h=[zeros(2,1);htheta(3:end,:)];

J=(1/(2*m))*sum((htheta-y).^2);

for t = 1:m
    yd = y(t);
    delta_o = htheta(t,1) - yd;
    delta_h_1 = who' * delta_o .*sigmoidgradient([1, s1(t, :)])';
    delta_h_1 = delta_h_1(2:end);
    delta_h_2 = who' * delta_o .*sigmoidgradient([1, s2(t, :)])';
    delta_h_2 = delta_h_2(2:end);
    delta_h_3 = who' * delta_o .*sigmoidgradient([1, s3(t, :)])';
    delta_h_3 = delta_h_3(2:end);
    delta_h_4 = who' * delta_o .*sigmoidgradient([1, s4(t, :)])';
    delta_h_4 = delta_h_4(2:end);
    delta_h_5 = who' * delta_o .*sigmoidgradient([1, s5(t, :)])';
    delta_h_5 = delta_h_5(2:end);
    

    wih_grad = wih_grad + delta_h_1 * ih(t, :);
    who_grad = who_grad + delta_o*hidden(t, :);
    wi1h_grad = wi1h_grad + delta_h_2 * i1h(t, :);
    wi2h_grad = wi2h_grad + delta_h_3 * i2h(t, :);
    wo1h_grad = wo1h_grad + delta_h_4 * o1h(t, :);
    wo2h_grad = wo2h_grad + delta_h_5 * o2h(t, :);
    
end

wih_grad = wih_grad / m;
who_grad = who_grad / m;
wi1h_grad = wi1h_grad / m;
wi2h_grad = wi2h_grad / m;
wo1h_grad = wo1h_grad / m;
wo2h_grad = wo2h_grad / m;

grad = [wih_grad(:) ; who_grad(:) ; wi1h_grad(:) ; wi2h_grad(:) ; wo1h_grad(:) ; wo2h_grad(:)];

end







