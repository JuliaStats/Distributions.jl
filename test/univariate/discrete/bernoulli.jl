using Distributions
using Test, Random

@test rand(Bernoulli()) isa Bool
@test rand(Bernoulli(), 10) isa Vector{Bool}
    
test_cgf(Bernoulli(0.5), (1f0, -1f0,1e6, -1e6))
test_cgf(Bernoulli(0.1), (1f0, -1f0,1e6, -1e6))
