function [norm,x_max,x_min]=maxminnormalise(x)

x_max=max(x); x_min=min(x);

norm=bsxfun(@minus,x,x_min);
norm=bsxfun(@rdivide,norm,(x_max-x_min));



end