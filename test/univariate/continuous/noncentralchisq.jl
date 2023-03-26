using Distributions
using Random
using Test

test_cgf(NoncentralChisq(3,2), (0.49, -1, -100, -1f6))
test_samples(NoncentralF(5, 10,5), 5000; rng=MersenneTwister(1234), nbins=50)

@testset "Native RNG" begin
    rng = MersenneTwister(1234)
    n = 5000
    d = NoncentralChisq(3, 2)
    samples = rand(rng, d, n)
    @test mean(samples) ≈ mean(d) atol=0.05*mean(d)
    @test var(samples) ≈ var(d) atol=0.1*var(d)
end
