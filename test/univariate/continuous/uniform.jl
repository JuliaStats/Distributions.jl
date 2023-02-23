using Distributions
using ChainRulesTestUtils
using OffsetArrays
using StatsFuns

using Random
using Test

@testset "uniform.jl" begin
    # affine transformations
    test_affine_transformations(Uniform, rand(), 4 + rand())
    test_cgf(Uniform(0,1),         (1, -1, 100f0, 1e6, -1e6))
    test_cgf(Uniform(100f0,101f0), (1, -1, 100f0, 1e6, -1e6))

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
    @testset "cgf uniform around 0" begin
        for (lo, hi, t) in [
            ((Float16(0), Float16(1), sqrt(eps(Float16)))),
            ((Float16(0), Float16(1), Float16(0))),
            ((Float16(0), Float16(1), -sqrt(eps(Float16)))),
            (0f0, 1f0, sqrt(eps(Float32))),
            (0f0, 1f0, 0f0),
            (0f0, 1f0, -sqrt(eps(Float32))),
            (-2f0, 1f0, 1f-30),
            (-2f-4, -1f-4, -2f-40),
            (0.0, 1.0, sqrt(eps(Float64))),
            (0.0, 1.0, 0.0),
            (0.0, 1.0, -sqrt(eps(Float64))),
            (-2.0, 5.0, -1e-35),
                           ]
            T = typeof(lo)
            @assert T == typeof(lo) == typeof(hi) == typeof(t)
            @assert t <= sqrt(eps(T))
            d = Uniform(lo, hi)
            precision = 512
            d_big = Uniform(BigFloat(lo, precision=precision), BigFloat(hi; precision=precision))
            t_big = BigFloat(t, precision=precision)
            @test cgf(d, t) isa T
            if iszero(t)
                @test cgf(d,t) === zero(t)
            else
                @test Distributions.cgf_around_zero(d, t) ≈ Distributions.cgf_away_from_zero(d_big, t_big) atol=eps(t) rtol=0
                @test Distributions.cgf_around_zero(d, t) === cgf(d, t)
            end
        end
    end
    # issue #1677
    @testset "consistency of pdf and cdf" begin
        for T in (Int, Float32, Float64)
            d = Uniform{T}(T(2), T(4))
            for S in (Int, Float32, Float64)
                TS = float(promote_type(T, S))

                @test @inferred(pdf(d, S(1))) === TS(0)
                @test @inferred(pdf(d, S(3))) === TS(1//2)
                @test @inferred(pdf(d, S(5))) === TS(0)

                @test @inferred(logpdf(d, S(1))) === TS(-Inf)
                @test @inferred(logpdf(d, S(3))) === -TS(logtwo)
                @test @inferred(logpdf(d, S(5))) === TS(-Inf)

                @test @inferred(cdf(d, S(1))) === TS(0)
                @test @inferred(cdf(d, S(3))) === TS(1//2)
                @test @inferred(cdf(d, S(5))) === TS(1)

                @test @inferred(logcdf(d, S(1))) === TS(-Inf)
                @test @inferred(logcdf(d, S(3))) === -TS(logtwo)
                @test @inferred(logcdf(d, S(5))) === TS(0)

                @test @inferred(ccdf(d, S(1))) === TS(1)
                @test @inferred(ccdf(d, S(3))) === TS(1//2)
                @test @inferred(ccdf(d, S(5))) === TS(0)

                @test @inferred(logccdf(d, S(1))) === TS(0)
                @test @inferred(logccdf(d, S(3))) === -TS(logtwo)
                @test @inferred(logccdf(d, S(5))) === TS(-Inf)
            end
        end
    end
end
