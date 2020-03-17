using Distributions
using Test, Random

@test rand(Bernoulli()) isa Bool
@test rand(Bernoulli(), 10) isa Vector{Bool}
