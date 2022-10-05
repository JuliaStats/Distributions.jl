using Distributions
using Test, Random

@test rand(BernoulliLogit()) isa Bool
@test rand(BernoulliLogit(), 10) isa Vector{Bool}
    
test_cgf(BernoulliLogit(), (1f0, -1f0, 1e6, -1e6))
test_cgf(BernoulliLogit(0.1), (1f0, -1f0, 1e6, -1e6))
