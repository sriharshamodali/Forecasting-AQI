function w = randomlyinitializeweights(in,out)

w = zeros(out,1 + in);

epsilon = 0.5;

w = rand(out, 1 + in) * 2 * epsilon - epsilon;

end
