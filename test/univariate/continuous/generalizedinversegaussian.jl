using Distributions
using Random

using Test

@testset "GeneralizedInverseGaussian" begin
    @testset "Constructors" begin
        # Argument checks
        p = randn()
        ab = -0.2
        @test_throws DomainError GeneralizedInverseGaussian(ab, 1.0, p)
        GeneralizedInverseGaussian(ab, 1.0, p; check_args=false)

        @test_throws DomainError GeneralizedInverseGaussian(1.0, ab, p)
        GeneralizedInverseGaussian(1.0, ab, p; check_args=false)
    end

    @testset "Special cases" begin
        a = Random.randexp()
        b = Random.randexp()
        # NOTE: p > 1 for Gamma to have a mode
        p_plus = Random.randexp() + 1
        p_minus = -Random.randexp()
        for (d, dref) in (
            (GeneralizedInverseGaussian(a, b, -1 // 2), InverseGaussian(sqrt(b / a), b)), # p = -1 // 2 (inverse gaussian)
            (GeneralizedInverseGaussian(a, 0.0, p_plus), Gamma(p_plus, 2 / a)), # b = 0 (gamma)
            (GeneralizedInverseGaussian(0.0, b, p_minus), InverseGamma(-p_minus, b / 2)) # a = 0 (inverse gamma)
        )
            @test minimum(d) == 0.0
            @test maximum(d) == Inf

            @test mean(d) ≈ mean(dref)

            # TODO: Uncomment after implementing `cdf` <30-04-23>
            # @test median(d) ≈ median(dref)
            @test mode(d) ≈ mode(dref)

            @test var(d) ≈ var(dref)
            @test std(d) ≈ std(dref)

            # TODO: Uncomment after implemented <30-04-23> 
            # @test kurtosis(d) ≈ kurtosis(dref) atol = 1e-12
            # @test entropy(d) ≈ entropy(dref)

            # PDF + CDF tests.
            for x in map((0.0, 1.0, 2.0, 3.0, 4.0f0)) do x
                # NOTE: InverseGamma is only defined for values greater than 0
                d isa InverseGamma ? (x == 0.0 ? x + eps() : x) : x
            end
                @test @inferred(pdf(d, x)) ≈ pdf(dref, x)

                # TODO: Uncomment after implemented <30-04-23> 
                @test @inferred(logpdf(d, x)) ≈ logpdf(dref, x) atol = 1e-6
                # @test @inferred(cdf(d, x)) ≈ cdf(dref, x) atol = 1e-12
                # @test @inferred(logcdf(d, x)) ≈ logcdf(dref, x) atol = 1e-12
            end
        end


        # TODO: Uncomment when implemented sampling <30-04-23> 
        # Additional tests, including sampling
        # test_distr(d, 10^6)
    end

    @testset "Non-special case" begin
        p = randn()
        a = Random.randexp()
        b = Random.randexp()
        d = GeneralizedInverseGaussian(a, b, p)

        @test minimum(d) == 0.0
        @test maximum(d) == Inf

        # FIX: Add true values <30-04-23> 
        # @test mean(d) == μ
        # @test median(d) == μ
        # @test mode(d) == μ

        # FIX: Add true values and uncomment when implemented <30-04-23> 
        # @test cdf(d, -Inf) == 0
        # @test logcdf(d, -Inf) == -Inf
        # @test cdf(d, μ) ≈ 0.5
        # @test logcdf(d, μ) ≈ -log(2)
        # @test cdf(d, Inf) == 1
        # @test logcdf(d, Inf) == 0
        # @test quantile(d, 1 // 2) ≈ μ

        # Additional tests, including sampling
        # test_distr(d, 10^6)
    end
end
