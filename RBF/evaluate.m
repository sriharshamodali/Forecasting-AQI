function predicted = evaluate(centers, betas, theta, input)

phi = RBFActivations(centers, betas, input); 

phi = phi ./ sum(phi); %normalising

phi = [1; phi];

predicted = theta' * phi;

end