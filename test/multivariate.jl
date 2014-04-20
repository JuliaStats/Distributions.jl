using Distributions
using Base.Test

mu = [1.0, 2.0, 3.0]
C = [4. -2. -1.; -2. 5. -1.; -1. -1. 6.]
h = mu
J = C
mv_test_distributions = [
	                 Dirichlet(3, 2.0), 
	                 Dirichlet([2.0, 1.0, 3.0]), 
	                 IsoNormal(mu, 2.0), 
	                 DiagNormal(mu, [1.5, 2.0, 2.5]), 
	                 MvNormal(mu, C), 
	                 IsoNormalCanon(h, 2.0), 
	                 DiagNormalCanon(h, [1.5, 2.0, 1.2]), 
	                 MvNormalCanon(h, J)]

for d in mv_test_distributions
    @test size(d) == size(rand(d))
    @test length(d) == length(rand(d))
end
