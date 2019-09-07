function IA_test=index_of_agreement(t,y)

t_avg=mean(t);

IA_test= 1-(sum((t-y).^2) / sum((abs(t-t_avg)+ abs(y-t_avg)).^2));


end