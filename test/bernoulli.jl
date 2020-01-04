using Distributions
using Test, Random

@test typeof(rand(Bernoulli())) == Int
@test typeof(rand(Bernoulli(), 10)) == Vector{Int}
