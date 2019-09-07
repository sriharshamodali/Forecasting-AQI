function p = predict_output(theta, x)

m = size(x, 1);
p = ([ones(m, 1) x] * theta);

end
