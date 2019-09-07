function r=correlation(x,y)

r= (mean(x.*y)-(mean(x)*mean(y)))/(std(x)*std(y));




end