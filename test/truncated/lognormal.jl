using Distributions, Test
using Distributions: expectation

naive_moment(d, n, μ, σ²) = (σ = sqrt(σ²); expectation(x -> ((x - μ) / σ)^n, d))

@testset "Truncated log normal" begin
    @testset "truncated(LogNormal{$T}(0, 1), ℯ⁻², ℯ²)" for T in (Float32, Float64, BigFloat)
        d = truncated(LogNormal{T}(zero(T), one(T)), exp(T(-2)), exp(T(2)))
        tn = truncated(Normal{BigFloat}(big(0.0), big(1.0)), -2, 2)
        bigmean = mgf(tn, 1)
        bigvar = mgf(tn, 2) - bigmean^2
        @test @inferred(mean(d)) ≈ bigmean
        @test @inferred(var(d)) ≈ bigvar
        @test @inferred(median(d)) ≈ one(T)
        @test @inferred(skewness(d)) ≈ naive_moment(d, 3, bigmean, bigvar)
        @test @inferred(kurtosis(d)) ≈ naive_moment(d, 4, bigmean, bigvar) - big(3)
        @test mean(d) isa T
    end
    @testset "Bound with no effect" begin
        # Uses the example distribution from issue #709, though what's tested here is
        # mostly unrelated to that issue (aside from `mean` not erroring).
        # The specified left truncation at 0 has no effect for `LogNormal`
        d1 = truncated(LogNormal(1, 5), 0, 1e5)
        @test mean(d1) ≈ 0 atol=eps()
        v1 = var(d1)
        @test v1 ≈ 0 atol=eps()
        # Without a `max(_, 0)`, this would be within machine precision of 0 (as above) but
        # numerically negative, which could cause downstream issues that assume a nonnegative
        # variance
        @test v1 >= 0
        # Compare results with not specifying a lower bound at all
        d2 = truncated(LogNormal(1, 5); upper=1e5)
        @test mean(d1) == mean(d2)
        @test var(d1) == var(d2)
    end
end
