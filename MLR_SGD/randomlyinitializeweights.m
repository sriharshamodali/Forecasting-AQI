function w = randomlyinitializeweights(in,out)

w = zeros(in+1,out);

epsilon = 0.5;

w = rand(in+1,out) * 2 * epsilon - epsilon;

end
