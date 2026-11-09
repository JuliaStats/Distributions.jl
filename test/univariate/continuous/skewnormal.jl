using Test
using Distributions

# Pointwise pdf/logpdf/cdf, moments, quantiles and sampling are verified against
# Azzalini's R package `sn` via the reference framework (see test/ref/continuous/skewnormal.R).
# This file covers construction, the support boundaries and the reduction to `Normal`.

@testset "SkewNormal" begin
    @test_throws DomainError SkewNormal(0.0, 0.0, 0.0)
    d1 = SkewNormal(1, 2, 3)
    d2 = SkewNormal(1.0f0, 2, 3)
    @test partype(d1) == Float64
    @test partype(d2) == Float32
    @test params(d1) == (1.0, 2.0, 3.0)
    @test minimum(d1) == -Inf
    @test maximum(d1) == Inf

    # support boundaries
    @test cdf(d1, -Inf) == 0
    @test cdf(d1, Inf) == 1
    @test ccdf(d1, -Inf) == 1
    @test ccdf(d1, Inf) == 0
    @test logcdf(d1, -Inf) == -Inf
    @test logccdf(d1, Inf) == -Inf
    @test quantile(d1, 0) == -Inf
    @test quantile(d1, 1) == Inf

    d0 = SkewNormal(0.0, 1.0, 0.0)
    @test SkewNormal() == d0

    # with alpha = 0 the skew normal reduces to the normal distribution
    d3 = SkewNormal(0.5, 2.2, 0.0)
    d4 = Normal(0.5, 2.2)
    @test pdf(d3, 3.3) == pdf(d4, 3.3)
    @test Base.Fix1(pdf, d3).(1:3) == Base.Fix1(pdf, d4).(1:3)
    @test cdf(d3, 3.3) ≈ cdf(d4, 3.3)
    @test ccdf(d3, 3.3) ≈ ccdf(d4, 3.3)
    @test (mean(d3), var(d3), std(d3)) == (mean(d4), var(d4), std(d4))
    @test skewness(d3) == skewness(d4)
    @test kurtosis(d3) == kurtosis(d4)
    @test mgf(d3, 2.25) == mgf(d4, 2.25)
    @test cf(d3, 2.25) == cf(d4, 2.25)
end
