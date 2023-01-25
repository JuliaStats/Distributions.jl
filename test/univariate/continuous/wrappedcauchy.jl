using Distributions
using Random: MersenneTwister
using Test

@testset "wrappedcauchy.jl" begin
    d = WrappedCauchy(0.6)

    @test params(d) == (0.6,)

    @test minimum(d) == -oftype(d.r,π)
    @test maximum(d) == oftype(d.r,π)
    @test extrema(d) == (-oftype(d.r,π), oftype(d.r,π))

    @test mean(d)     ==  .0
    @test var(d)      == 0.4
    @test skewness(d) ==  .0
    @test median(d)   ==  .0
    @test mode(d)     ==  .0
    @test entropy(d)  == 1.33787706640934544458

    @test pdf(d, -10) == .0
    @test pdf(d, 2π) == .0
    #@test pdf(d,  0) == .31830988618379067154

    @test logpdf(d, -10) == -Inf
    @test logpdf(d, 2π) == -Inf
    #@test logpdf(d,  0) ≈  log(.31830988618379067154)

    @test cdf(d, -10) ==  .0
    @test cdf(d, 2π) ==  1.0
    @test cdf(d,  0) ==  .5

    @test quantile(d,  .0) == -oftype(d.r,π)
    @test quantile(d,  .5) ==   .0
    @test quantile(d, 1.0) == +oftype(d.r,π)
end
