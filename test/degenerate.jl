using Test, Distributions, StatsBase

@testset "Degenerate Beta" begin
    d1 = Beta(.5, Inf)
    d2 = Beta(Inf, Inf)
    d3 = Beta(Inf, 14)

    @test minimum(d1) == 1
    @test minimum(d2) == .5
    @test minimum(d3) == 0

    @test maximum(d1) == 1
    @test maximum(d2) == .5
    @test maximum(d3) == 0

    @test mean(d1) == 1
    @test mean(d2) == 5
    @test mean(d3) == 0

    # Currently hangs due to StatsFuns
    # @test median(d1) == 1
    # @test median(d2) == .5
    # @test median(d3) == 0

    @test mode(d1) == 1
    @test mode(d2) == .5
    @test mode(d3) == 0

    @test var(d1) == 0
    @test var(d2) == 0
    @test var(d3) == 0

    @test std(d3) == 0

    @test isnan(skewness(d1))
    @test isnan(skewness(d2))
    @test isnan(skewness(d3))

    @test isnan(kurtosis(d1))
    @test isnan(kurtosis(d2))
    @test isnan(kurtosis(d3))

    @test entropy(d1) == 0
    @test entropy(d2) == 0
    @test entropy(d3) == 0

    @test meanlogx(d1) == 0
    @test meanlogx(d2) == log(.5)
    @test meanlogx(d3) == log(0)

    @test varlogx(d1) == 0
    @test varlogx(d2) == 0
    @test varlogx(d3) == 0

    @test stdlogx(d1) == 0

    @test rand(d1) == 1
    @test rand(d2) == .5
    @test rand(d3) == 0

    @test rand(sampler(d1)) == 1
    @test rand(sampler(d2)) == .5
    @test rand(sampler(d3)) == 0
end
