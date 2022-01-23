using Distributions
using ChainRulesTestUtils
using Test

@testset "uniform.jl" begin
    # affine transformations
    test_affine_transformations(Uniform, rand(), 4 + rand())

    @testset "loglikelihood" begin
        dist = Uniform(rand(), 2 + rand())
        for dims in ((10,), (2, 5), (1, 2, 3, 1))
            # all values in support
            x = rand(dist, dims...)
            @test @inferred(loglikelihood(dist, x)) ≈ sum(logpdf.(dist, x))

            # one value not in support
            x[rand(1:length(x))] = -1
            @test @inferred(loglikelihood(dist, x)) == -Inf
        end
    end

    @testset "ChainRules" begin
        # run test suite for values in the support
        dist = Uniform(- 1 - rand(), 1 + rand())
        tangent = ChainRulesTestUtils.Tangent{Uniform{Float64}}(; a=randn(), b=randn())
        for x in (rand(), -rand())
            test_frule(logpdf, dist ⊢ tangent, x)
            test_rrule(logpdf, dist ⊢ tangent, x)
        end
        for dim in ((10,), (2, 5), (1, 2, 3, 1))
            x = 2 .* rand(dim...) .- 1
            test_frule(loglikelihood, dist ⊢ tangent, x)
            test_rrule(loglikelihood, dist ⊢ tangent, x)
        end

        # check manually that otherwise derivatives are zero (FiniteDifferences returns NaN)
        for x in (-2, 2)
            # frule
            @test @inferred(
                ChainRulesTestUtils.frule(
                    (ChainRulesTestUtils.NoTangent(), tangent, randn()),
                    logpdf,
                    dist,
                    x,
                ),
            ) == (-Inf, 0.0)

            # rrule
            Ω, pullback = @inferred(ChainRulesTestUtils.rrule(logpdf, dist, x))
            @test Ω == -Inf
            @test @inferred(pullback(randn())) == (
                ChainRulesTestUtils.NoTangent(),
                ChainRulesTestUtils.Tangent{Uniform{Float64}}(; a=0.0, b=0.0),
                ChainRulesTestUtils.ZeroTangent(),
            )
        end
        for x in (vcat(-2, rand(dist, 9)), vcat(rand(dist, 9), 2))
            # frule
            @test @inferred(
                ChainRulesTestUtils.frule(
                    (ChainRulesTestUtils.NoTangent(), tangent, randn()),
                    loglikelihood,
                    dist,
                    x,
                ),
            ) == (-Inf, 0.0)

            # rrule
            Ω, pullback = @inferred(ChainRulesTestUtils.rrule(loglikelihood, dist, x))
            @test Ω == -Inf
            @test @inferred(pullback(randn())) == (
                ChainRulesTestUtils.NoTangent(),
                ChainRulesTestUtils.Tangent{Uniform{Float64}}(; a=0.0, b=0.0),
                ChainRulesTestUtils.ZeroTangent(),
            )
        end
    end
end
