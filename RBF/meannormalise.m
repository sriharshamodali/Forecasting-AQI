function [x,avg,sd]=meannormalise(x)

avg = zeros(1, size(x, 2));
sd = zeros(1, size(x, 2));
avg = mean(x);
sd = std(x);
for i = 1:size(x,2)
    x(:,i) = (x(:,i) - avg(i)) / sd(i);
end

end