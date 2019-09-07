function z = RBFActivations(centers, betas, input)

    diff = bsxfun(@minus, centers, input);
    sqrddists = sum(diff .^ 2, 2);
    z = exp(-betas .* sqrddists);
    %z=sqrt(1+betas.*sqrddists);
    %z=(1+sqrddists).^(betas/2);
    

end