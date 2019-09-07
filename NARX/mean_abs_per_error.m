function MAPE=mean_abs_per_error(t,y)


err = abs(bsxfun(@minus, y, t));
pcterr = bsxfun(@rdivide, err, t);
MAPE = nanmean(pcterr,1);
MAPE=MAPE*100;

end