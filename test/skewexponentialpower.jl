using Test
using Distributions

@testset "SkewExponentialPower" begin
    @test_throws ArgumentError SkewExponentialPower(0, 0, 0, 0)
    d1 = SkewExponentialPower(0, 1, 1, 0.5f0)
    @test partype(d1) == Float32
    d2 = SkewExponentialPower(0, 1, 1, 0.5)
    @test partype(d2) == Float64
    @test params(d2) == (0., 1., 1., 0.5)
    @test cdf(d2, Inf) == 1
    @test cdf(d2, -Inf) == 0
    @test quantile(d2, 1) == Inf
    @test quantile(d2, 0) == -Inf

    # Comparison to laplace
    d = SkewExponentialPower(0, 1, 1, 0.5)
    dl = Laplace(0, 1)
    @test mean(d) ≈ mean(dl)
    @test var(d) ≈ var(dl)
    @test skewness(d) ≈ skewness(dl)
    @test kurtosis(d) ≈ kurtosis(dl)
    @test pdf(d, 0.5) ≈ pdf(dl, 0.5)
    @test cdf(d, 0.5) ≈ cdf(dl, 0.5)
    @test quantile(d, 0.5) ≈ quantile(dl, 0.5)

    # comparison to exponential power distribution (PGeneralizedGaussian),
    # where the variance is reparametrized as σₚ = p^(1/p)σ to ensure equal pdfs
    p = 1.2
    d = SkewExponentialPower(0, 1, p, 0.5)
    de = PGeneralizedGaussian(0, p^(1/p), p)
    @test mean(d) ≈ mean(de)
    @test var(d) ≈ var(de)
    @test skewness(d) ≈ skewness(de)
    @test kurtosis(d) ≈ kurtosis(de)
    @test pdf(d, 0.5) ≈ pdf(de, 0.5)
    @test cdf(d, 0.5) ≈ cdf(de, 0.5)

    # Comparison to normal
    d = SkewExponentialPower(0, 1, 2, 0.5)
    dn = Normal(0, 1)
    @test mean(d) ≈ mean(dn)
    @test var(d) ≈ var(dn)
    @test skewness(d) ≈ skewness(dn)
    @test kurtosis(d) ≈ kurtosis(dn)
    @test pdf(d, 0.5) ≈ pdf(dn, 0.5)
    @test cdf(d, 0.5) ≈ cdf(dn, 0.5)
    @test quantile(d, 0.5) ≈ quantile(dn, 0.5)
end
