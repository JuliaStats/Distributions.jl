using Distributions
using Test, Random

@test rand(Bernoulli()) isa Bool
@test typeof(rand(Bernoulli(), 10)) == Vector{Int}
