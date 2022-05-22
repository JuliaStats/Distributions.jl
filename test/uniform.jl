using Distributions
using ChainRulesTestUtils
using OffsetArrays

using Random
using Test

@testset "uniform.jl" begin
    # affine transformations
    test_affine_transformations(Uniform, rand(), 4 + rand())

    @testset "ChainRules" begin
        # run test suite for values in the support
        dist = Uniform(- 1 - rand(), 1 + rand())
        tangent = ChainRulesTestUtils.Tangent{Uniform{Float64}}(; a=randn(), b=randn())
        for x in (rand(), -rand())
            test_frule(logpdf, dist ⊢ tangent, x)
            test_rrule(logpdf, dist ⊢ tangent, x)
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
    end

    @testset "fit: array indexing (#1253)" begin
        x = shuffle(10:20)
        for data in (x, OffsetArray(x, -5:5))
            @test fit(Uniform, data) == Uniform(10, 20)
        end
    end
end
