using Distributions
using Random

using Test

@testset "GeneralizedInverseGaussian" begin
    @testset "Constructors" begin
        # Argument checks
        λ = randn()
        ψχ = -0.2
        @test_throws DomainError GeneralizedInverseGaussian(λ, ψχ, 1.0)
        GeneralizedInverseGaussian(λ, ψχ, 1.0; check_args=false) # passes

        @test_throws DomainError GeneralizedInverseGaussian(λ, 1.0, ψχ)
        GeneralizedInverseGaussian(λ, 1.0, ψχ; check_args=false) # passes
    end

    @testset "Special cases" begin
        ψ = Random.randexp()
        χ = Random.randexp()
        # NOTE: λ > 1 for Gamma to have a mode
        λ_plus = Random.randexp() + 1
        λ_minus = -Random.randexp()
        for (d, dref) in (
            (GeneralizedInverseGaussian(-1 // 2, ψ, χ), InverseGaussian(sqrt(χ / ψ), χ)), # λ = -1 // 2 => inverse gaussian)
            (GeneralizedInverseGaussian(λ_plus, ψ, 0.0), Gamma(λ_plus, 2 / ψ)),           # χ = 0 => gamma
            (GeneralizedInverseGaussian(λ_minus, 0.0, χ), InverseGamma(-λ_minus, χ / 2))  # ψ = 0 => inverse gamma
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
        λ = randn()
        ψ = Random.randexp()
        χ = Random.randexp()
        d = GeneralizedInverseGaussian(λ, ψ, χ)

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
