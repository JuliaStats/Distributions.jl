using Distributions
using SpecialFunctions

using Test

@testset "AsymmetricExponentialPower" begin
    @testset "p₁ = p₂" begin
        @testset "α = 0.5" begin
            @test_throws DomainError AsymmetricExponentialPower(0, 0, 0, 0, 0)
            d1 = AsymmetricExponentialPower(0, 1, 1, 1, 0.5f0)
            @test @inferred partype(d1) == Float32
            d2 = AsymmetricExponentialPower(0, 1, 1, 1, 0.5)
            @test @inferred partype(d2) == Float64
            @test @inferred params(d2) == (0., 1., 1., 1., 0.5)
            @test @inferred cdf(d2, Inf) == 1
            @test @inferred cdf(d2, -Inf) == 0
            @test @inferred quantile(d2, 1) == Inf
            @test @inferred quantile(d2, 0) == -Inf
            test_distr(d2, 10^6)

            # Comparison to SEPD
            d = AsymmetricExponentialPower(0, 1, 1, 1, 0.5)
            dl = SkewedExponentialPower(0, 1, 1, 0.5)
            @test @inferred mean(d) ≈ mean(dl)
            @test @inferred var(d) ≈ var(dl)
            @test @inferred skewness(d) ≈ skewness(dl)
            @test @inferred kurtosis(d) ≈ kurtosis(dl)
            @test @inferred pdf(d, 0.5) ≈ pdf(dl, 0.5)
            @test @inferred cdf(d, 0.5) ≈ cdf(dl, 0.5)
            @test @inferred quantile(d, 0.5) ≈ quantile(dl, 0.5)

            # Comparison to laplace
            d = AsymmetricExponentialPower(0, 1, 1, 1, 0.5)
            dl = Laplace(0, 1)
            @test @inferred mean(d) ≈ mean(dl)
            @test @inferred var(d) ≈ var(dl)
            @test @inferred skewness(d) ≈ skewness(dl)
            @test @inferred kurtosis(d) ≈ kurtosis(dl)
            @test @inferred pdf(d, 0.5) ≈ pdf(dl, 0.5)
            @test @inferred cdf(d, 0.5) ≈ cdf(dl, 0.5)
            @test @inferred quantile(d, 0.5) ≈ quantile(dl, 0.5)

            # comparison to exponential power distribution (PGeneralizedGaussian),
            # where the variance is reparametrized as σₚ = p^(1/p)σ to ensure equal pdfs
            p = 1.2
            d = AsymmetricExponentialPower(0, 1, p, p, 0.5)
            de = PGeneralizedGaussian(0, p^(1/p), p)
            @test @inferred mean(d) ≈ mean(de)
            @test @inferred var(d) ≈ var(de)
            @test @inferred skewness(d) ≈ skewness(de)
            @test @inferred kurtosis(d) ≈ kurtosis(de)
            @test @inferred pdf(d, 0.5) ≈ pdf(de, 0.5)
            @test @inferred cdf(d, 0.5) ≈ cdf(de, 0.5)
            test_distr(d, 10^6)
        end

        @testset "α != 0.5" begin
            # relationship between aepd(μ, σ, p, p, α) and
            # aepd(μ, σ, p, p, 1-α)
            d1 = AsymmetricExponentialPower(0, 1, 0.1, 0.1, 0.7)
            d2 = AsymmetricExponentialPower(0, 1, 0.1, 0.1, 0.3)
            @inferred -mean(d1) ≈ mean(d2)
            @inferred var(d1) ≈ var(d2)
            @inferred -skewness(d1) ≈ skewness(d2)
            @inferred kurtosis(d1) ≈ kurtosis(d2)

            α, p = rand(2)
            d = SkewedExponentialPower(0, 1, p, α)
            # moments of the standard SEPD, Equation 18 in Zhy, D. and V. Zinde-Walsh (2009)
            moments = [(2*p^(1/p))^k * ((-1)^k*α^(1+k) + (1-α)^(1+k)) * gamma((1+k)/p)/gamma(1/p) for k ∈ 1:4]

            @inferred var(d) ≈ moments[2] - moments[1]^2
            @inferred skewness(d) ≈ moments[3] / (√(moments[2] - moments[1]^2))^3
            @inferred kurtosis(d) ≈ (moments[4] / ((moments[2] - moments[1]^2))^2 - 3)
            test_distr(d, 10^6)
        end
    end
    @testset "p₁ ≠ p₂" begin
        d2 = AsymmetricExponentialPower(0, 1., 1., 1., 0.7)
        test_distr(d2, 10^6)

        # relationship between aepd(0, σ, p₁, p₂, α) and aepd(0, σ, p₂, p₁, 1-α)
        d1 = AsymmetricExponentialPower(0, 1., 0.4, 1.2, 0.7)
        d2 = AsymmetricExponentialPower(0, 1., 1.2, 0.4, 0.3)
        @inferred -mean(d1) ≈ mean(d2)
        @inferred var(d1) ≈ var(d2)
        @inferred -skewness(d1) ≈ skewness(d2)
        @inferred kurtosis(d1) ≈ kurtosis(d2)
    end
end