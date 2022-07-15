using Distributions
using ChainRulesTestUtils
using OffsetArrays

using Random
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
        # run test suite for values inside the support
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
        for x in (-2, dist.a, dist.b, 2)
            # frule
            @test @inferred(
                ChainRulesTestUtils.frule(
                    (ChainRulesTestUtils.NoTangent(), tangent, randn()),
                    logpdf,
                    dist,
                    x,
                ),
            ) == (logpdf(dist, x), 0.0)

            # rrule
            Ω, pullback = @inferred(ChainRulesTestUtils.rrule(logpdf, dist, x))
            @test Ω == logpdf(dist, x)
            @test @inferred(pullback(randn())) == (
                ChainRulesTestUtils.NoTangent(),
                ChainRulesTestUtils.Tangent{Uniform{Float64}}(; a=0.0, b=0.0),
                ChainRulesTestUtils.ZeroTangent(),
            )
        end
        for x in (
            fill(-2, 10),
            fill(dist.a, 2, 5),
            fill(dist.b, 1, 1, 4),
            fill(2, 3, 2, 1),
            shuffle!(vcat(-2, rand(9))),
            shuffle!(vcat(dist.a, 2, rand(8))),
            shuffle!(vcat(dist.b, -2, 2, rand(6))),
        )
            # frule
            ΔΩ = sum(x) do xi
                _, ΔΩi = ChainRulesTestUtils.frule(
                    (ChainRulesTestUtils.NoTangent(), tangent, randn()),
                    logpdf,
                    dist,
                    xi,
                )
                return ΔΩi
            end
            ChainRulesTestUtils.test_approx(
                @inferred(
                    ChainRulesTestUtils.frule(
                        (ChainRulesTestUtils.NoTangent(), tangent, randn()),
                        loglikelihood,
                        dist,
                        x,
                    ),
                ),
                (loglikelihood(dist, x), ΔΩ),
            )

            # rrule
            pullbacks = map(x) do xi
                _, pullback = ChainRulesTestUtils.rrule(logpdf, dist, xi)
                return pullback
            end
            Ω, pullback = @inferred(ChainRulesTestUtils.rrule(loglikelihood, dist, x))
            @test Ω == loglikelihood(dist, x)
            Δ = randn()
            Δd = sum(pullbacks) do pb
                _, Δdi, _ = pb(Δ)
                return Δdi
            end
            @assert Δd isa ChainRulesTestUtils.Tangent{Uniform{Float64}}
            ChainRulesTestUtils.test_approx(
                @inferred(pullback(Δ)),
                (ChainRulesTestUtils.NoTangent(), Δd, ChainRulesTestUtils.ZeroTangent()),
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
