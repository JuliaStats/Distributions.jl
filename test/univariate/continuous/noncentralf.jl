using Distributions
using Random
using Test

test_samples(NoncentralF(5, 10,5), 5000; rng=MersenneTwister(1234), nbins=60)

@testset "Native RNG" begin
    rng = MersenneTwister(1234)
    nsamples = 5000
    d = NoncentralF(5, 10, 2)
    samples = rand(rng, d, nsamples)
    @test mean(samples) ≈ mean(d) atol=0.05*mean(d)
end