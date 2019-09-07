function norm=maxminnormalise(x)
[m,n]=size(x); norm=zeros(m,n);
for i=1:size(x,2)
   norm(:,i)=(x(:,i)-min(x(:,i)))/(max(x(:,i))-min(x(:,i))); 
end


end