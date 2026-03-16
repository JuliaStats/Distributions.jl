@testset "Tweedie quantile bounds" begin
    d = Tweedie(2.0, 1.5, 1.0)
    @test quantile(d, 0.0) == 0.0
    @test quantile(d, 1.0) == Inf
end

@testset "Tweedie pdf" begin
    d = Tweedie(2.0, 1.5, 1.0)
    @test logpdf(d, -0.3) == -Inf
    @test pdf(d, -0.3) == 0
    @test logpdf(d, -0.0) == logpdf(d, 0.0)
    @test pdf(d, -0.0) == pdf(d, 0.0)
end

@testset "Tweedie overflows" begin
    d = Tweedie(1.1, 1.0, 1.01)
    # This case overflows when using `wrightbessel` rather than `logwrightbessel`
    @test pdf(d, 10) ≈ 3.1785795511027634e-7
    # This one would return Inf if we don't manually return NaN
    @test isnan(pdf(d, 15))
end

@testset "Tweedie custom type" begin
    d1 = Tweedie(2.0, 1.5, 1.0)
    d2 = Tweedie(2.0f0, 1.5f0, 1.0f0)
    @test pdf(d2, 1)::Float32 ≈ pdf(d1, 1)
    @test cdf(d2, 1)::Float32 ≈ cdf(d1, 1)
    @test quantile(d2, 0.1)::Float32 ≈ quantile(d1, 0.1)
    @test mean(d2)::Float32 ≈ mean(d1)
    @test median(d2)::Float32 ≈ median(d1)
    @test skewness(d2)::Float32 ≈ skewness(d1)
    @test kurtosis(d2)::Float32 ≈ kurtosis(d1)
end

@testset "Tweedie (c)quantile and invlog(c)cdf" begin
    # check that rounding from exact computation to floating point
    # does not break round-tripping due to mass at zero
    d = Tweedie(1.1, 1, 1.01)
    @test quantile(d, cdf(d, 0)) == 0
    @test cquantile(d, ccdf(d, 0)) == 0
    @test invlogcdf(d, logcdf(d, 0)) == 0
    @test invlogccdf(d, logccdf(d, 0)) == 0

    @test_throws DomainError cquantile(d, -0.1)
    @test_throws DomainError cquantile(d, 1.1)
end