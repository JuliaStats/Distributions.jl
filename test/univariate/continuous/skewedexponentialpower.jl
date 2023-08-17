using Distributions
using SpecialFunctions

using Test

@testset "SkewedExponentialPower" begin
    @testset "α = 0.5" begin
        @test_throws DomainError SkewedExponentialPower(0, 0, 0, 0)
        d1 = SkewedExponentialPower(0, 1, 1, 0.5f0)
        @test @inferred partype(d1) == Float32
        d2 = SkewedExponentialPower(0, 1, 1, 0.5)
        @test @inferred partype(d2) == Float64
        @test @inferred params(d2) == (0., 1., 1., 0.5)
        @test @inferred cdf(d2, Inf) == 1
        @test @inferred cdf(d2, -Inf) == 0
        @test @inferred quantile(d2, 1) == Inf
        @test @inferred quantile(d2, 0) == -Inf
        test_distr(d2, 10^6)

        # Comparison to laplace
        d = SkewedExponentialPower(0, 1, 1, 0.5)
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
        d = SkewedExponentialPower(0, 1, p, 0.5)
        de = PGeneralizedGaussian(0, p^(1/p), p)
        @test @inferred mean(d) ≈ mean(de)
        @test @inferred var(d) ≈ var(de)
        @test @inferred skewness(d) ≈ skewness(de)
        @test @inferred kurtosis(d) ≈ kurtosis(de)
        @test @inferred pdf(d, 0.5) ≈ pdf(de, 0.5)
        @test @inferred cdf(d, 0.5) ≈ cdf(de, 0.5)
        test_distr(d, 10^6)

        # This is infinite for the PGeneralizedGaussian implementation
        d = SkewedExponentialPower(0, 1, 0.01, 0.5)
        @test @inferred isfinite(var(d))
        test_distr(d, 10^6)

        # Comparison to normal
        d = SkewedExponentialPower(0, 1, 2, 0.5)
        dn = Normal(0, 1)
        @test @inferred mean(d) ≈ mean(dn)
        @test @inferred var(d) ≈ var(dn)
        @test @inferred skewness(d) ≈ skewness(dn)
        @test @inferred isapprox(kurtosis(d), kurtosis(dn), atol = 1/10^10)
        @test @inferred pdf(d, 0.5) ≈ pdf(dn, 0.5)
        @test @inferred cdf(d, 0.5) ≈ cdf(dn, 0.5)
        @test @inferred quantile(d, 0.5) ≈ quantile(dn, 0.5)
        test_distr(d, 10^6)
    end
    @testset "α != 0.5" begin
        # Format is [x, pdf, cdf] from the asymmetric
        # exponential power function in R from package
        # VaRES. Values are set to μ = 0, σ = 1, p = 0.5, α = 0.7
        test = [
            -10.0000000 0.004770878 0.02119061;
            -9.2631579 0.005831228 0.02508110;
            -8.5263158 0.007185598 0.02985598;
            -7.7894737 0.008936705 0.03576749;
            -7.0526316 0.011232802 0.04315901;
            -6.3157895 0.014293449 0.05250781;
            -5.5789474 0.018454000 0.06449163;
            -4.8421053 0.024246394 0.08010122;
            -4.1052632 0.032555467 0.10083623;
            -3.3684211 0.044947194 0.12906978;
            -2.6315789 0.064438597 0.16879238;
            -1.8947368 0.097617334 0.22732053;
            -1.1578947 0.162209719 0.32007314;
            -0.4210526 0.333932302 0.49013645;
            0.3157895 0.234346966 0.82757893;
            1.0526316 0.070717323 0.92258764;
            1.7894737 0.031620241 0.95773428;
            2.5263158 0.016507946 0.97472202;
            3.2631579 0.009427166 0.98399211;
            4.0000000 0.005718906 0.98942757;
        ]

        d = SkewedExponentialPower(0, 1, 0.5, 0.7)
        for t in eachrow(test)
            @test @inferred(pdf(d, t[1])) ≈ t[2] rtol=1e-5
            @test @inferred(logpdf(d, t[1])) ≈ log(t[2]) rtol=1e-5
            @test @inferred(cdf(d, t[1])) ≈ t[3] rtol=1e-3
            @test @inferred(logcdf(d, t[1])) ≈ log(t[3]) rtol=1e-3
        end
        test_distr(d, 10^6)

        # relationship between sepd(μ, σ, p, α) and
        # sepd(μ, σ, p, 1-α)
        d1 = SkewedExponentialPower(0, 1, 0.1, 0.7)
        d2 = SkewedExponentialPower(0, 1, 0.1, 0.3)
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
