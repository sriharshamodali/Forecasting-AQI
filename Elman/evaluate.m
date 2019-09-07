function predicted=evaluate(x,wih,wch,who,input_layer_size,hidden_layer_size,context_layer_size)

m = size(x, 1); 
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
predicted=sigmoid(hidden*who');


end