function MAE_test=mean_abs_err(t,y)

m=length(y);

MAE_test= (1/m)*sum(abs(t-y));

end
