function pred = predict_out(x,wih,who,wi1h,wi2h,wo1h,wo2h,hidden_layer_size)

m = size(x, 1);
pred=zeros(m,1);
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
    pred(k)=sigmoid(s6(k,:));
    s4(k+1,:)=pred(k,:)*wo1h';
    s5(k+2,:)=pred(k,:)*wo2h';
end

end