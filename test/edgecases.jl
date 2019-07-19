using Test, Distributions, StatsBase

@testset "TDist(Inf) is Normal(0,1)" begin
    T = TDist(Inf)
    N = Normal(0, 1)

    x = rand(N)
    z = rand(T, 10000000)

    @test mean(T) == mean(N)
    @test median(T) == median(N)
    @test mode(T) == mode(N)
    @test var(T) == var(N)
    @test skewness(T) == skewness(N)
    @test kurtosis(T) == kurtosis(N)
    @test entropy(T) == entropy(N)
    @test pdf(T, x) ≈ pdf(N, x)
    @test logpdf(T, x) ≈ logpdf(N, x)
    @test gradlogpdf(T, x) ≈ gradlogpdf(N, x)
    @test cf(T, x) ≈ cf(N, x)

    fnecdf = ecdf(z)
    y = [-1.96, -1.644854, -1.281552, -0.6744898, 0, 0.6744898, 1.281552, 1.644854, 1.96]
    @test isapprox(fnecdf(y), [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975], atol=1e-3)
end
