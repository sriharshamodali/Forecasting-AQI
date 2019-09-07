function w = randomlyinitializeweights(in,out)

w = zeros(in,out);

epsilon = 0.5;

w = rand(in,out) * 2 * epsilon - epsilon;

end
