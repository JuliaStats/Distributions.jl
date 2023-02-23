using Distributions

using Random
using Test

@testset "PGeneralizedGaussian" begin
    @testset "Constructors" begin
        # Argument checks
        μ = randn()
        for p in (-0.2, 0)
            @test_throws DomainError PGeneralizedGaussian(p)
            PGeneralizedGaussian(p; check_args=false)

            @test_throws DomainError PGeneralizedGaussian(μ, 1.0, p)
            PGeneralizedGaussian(μ, 1.0, p; check_args=false)
        end
        for α in (-1.2, 0)
            @test_throws DomainError PGeneralizedGaussian(μ, α, 1.0)
            PGeneralizedGaussian(μ, α, 1.0; check_args=false)

            for p in (-0.2, 0)
                @test_throws DomainError PGeneralizedGaussian(μ, α, p)
                PGeneralizedGaussian(μ, α, p; check_args=false)
            end
        end

        # Convenience constructors
        d = PGeneralizedGaussian()
        @test d.μ == 0
        @test d.α ≈ sqrt(2)
        @test d.p == 2

        d = PGeneralizedGaussian(2.1)
        @test d.μ == 0
        @test d.α == 1
        @test d.p == 2.1
    end

    @testset "Special cases" begin
        μ = randn()
        α = Random.randexp()
        for (d, dref) in (
            (PGeneralizedGaussian(μ, α, 1), Laplace(μ, α)), # p = 1 (Laplace)
            (PGeneralizedGaussian(), Normal()), # p = 2 (standard normal)
            (PGeneralizedGaussian(μ, α, 2), Normal(μ, α / sqrt(2))), # p = 2 (normal)
        )
            @test minimum(d) == -Inf
            @test maximum(d) == Inf

            @test location(d) == d.μ
            @test scale(d) == d.α
            @test shape(d) == d.p

            @test mean(d) == d.μ
            @test mean(d) ≈ mean(dref)
            @test median(d) == d.μ
            @test median(d) ≈ median(dref)
            @test mode(d) == d.μ
            @test mode(d) ≈ mode(dref)

            @test var(d) ≈ var(dref)
            @test std(d) ≈ std(dref)

            @test skewness(d) == 0
            @test kurtosis(d) ≈ kurtosis(dref) atol = 1e-12
            @test entropy(d) ≈ entropy(dref)

            # PDF + CDF tests.
            for x in (-Inf, d.μ - 4.2, d.μ - 1.2, d.μ, Float32(d.μ) + 0.3f0, d.μ + 4, Inf32)
                @test @inferred(pdf(d, x)) ≈ pdf(dref, x)
                @test @inferred(logpdf(d, x)) ≈ logpdf(dref, x)
                @test @inferred(cdf(d, x)) ≈ cdf(dref, x) atol = 1e-12
                @test @inferred(logcdf(d, x)) ≈ logcdf(dref, x) atol = 1e-12
            end

            # Additional tests, including sampling
            test_distr(d, 10^6)
        end
    end

    @testset "Non-special case" begin
        μ = randn()
        α = Random.randexp()
        p = Random.randexp()
        d = PGeneralizedGaussian(μ, α, p)

        @test minimum(d) == -Inf
        @test maximum(d) == Inf

        @test location(d) == μ
        @test scale(d) == α
        @test shape(d) == p

        @test mean(d) == μ
        @test median(d) == μ
        @test mode(d) == μ

        @test cdf(d, -Inf) == 0
        @test logcdf(d, -Inf) == -Inf
        @test cdf(d, μ) ≈ 0.5
        @test logcdf(d, μ) ≈ -log(2)
        @test cdf(d, Inf) == 1
        @test logcdf(d, Inf) == 0
        @test quantile(d, 1 // 2) ≈ μ

        # Additional tests, including sampling
        test_distr(d, 10^6)
    end
end
