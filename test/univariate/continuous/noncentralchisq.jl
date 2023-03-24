using Distributions
using Random
using Test

@testset "Native RNG" begin
    rng = MersenneTwister(1234)
    n = 5000
    d = NoncentralChisq(3, 2)
    samples = rand(rng, d, n)
    @test mean(samples) ≈ mean(d) atol=0.05*mean(d)
    @test var(samples) ≈ var(d) atol=0.1*var(d)
end
