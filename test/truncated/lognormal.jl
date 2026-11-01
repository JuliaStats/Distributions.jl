using Distributions
using ForwardDiff
using SpecialFunctions
using Test

# Mean and variance of LogNormal(μ, σ) truncated below (`side = :lower`) or above
# (`side = :upper`) at x, computed directly from E[X^k] in high precision
function tlognormal_moments_big(μ, σ, x, side::Symbol)
    return setprecision(BigFloat, 512) do
        μ, σ, x = big(μ), big(σ), big(x)
        Q(z) = erfc(z / sqrt(big(2))) / 2
        z = (log(x) - μ) / σ
        # E[X^k] = exp(kμ + k²σ²/2) Q(sz - skσ) / Q(sz) with s = 1 below and s = -1 above
        s = side === :lower ? 1 : -1
        m1 = exp(μ + σ^2 / 2) * Q(s * z - s * σ) / Q(s * z)
        m2 = exp(2μ + 2σ^2) * Q(s * z - 2s * σ) / Q(s * z)
        return m1, m2 - m1^2
    end
end

@testset "Truncated log-normal, mean and variance" begin
    @testset "reference values" begin
        # independent 80-digit values for X | X ≥ 1, X ~ LogNormal(μ, σ), with decimal inputs
        for (μ, σ, m, v) in (
            (-3.0, 0.2, big"1.0133936271158251321640304503608421030925238110513", big"0.00018262102728365573837368596625657316779182460453137"),
            (0.0, 0.2, big"1.1819230635865642306502376887506657517532908353084", big"0.023077664881075315916880212629814083418221748756613"),
            (3.0, 0.8, big"27.662722228075855013122702422460104896062130123345", big"685.8900639971743347068289425610521604132805106209"),
            (-40.0, 1.0, big"1.0256078378620389719011077557310990700258820858773", big"0.00068930271665356090955506298758264910853003901221569"),
        )
            d = truncated(LogNormal(μ, σ); lower=1)
            @test mean(d) ≈ m rtol=1e-14
            @test var(d) ≈ v rtol=1e-14
            @test std(d) ≈ sqrt(v) rtol=1e-14
        end
    end

    # standardized bounds from far below to far above the median and scales from narrow to wide;
    # the accuracy does not depend on cancellation between the second moment and the squared mean
    @testset "accuracy, $side bound" for side in (:lower, :upper)
        for z in (-40.0, -5.0, -1.0, 0.0, 0.5, 1.5, 2.0, 2.5, 4.0, 10.0, 15.0, 40.0, 1e3),
            σ in (1e-8, 1e-4, 1e-2, 0.2, 0.5, 1.0, 2.0, 5.0)
            # the bound x = 1 lies z standard deviations into the upper (lower) tail for a lower
            # (upper) bound
            μ = side === :lower ? -z * σ : z * σ
            m_ref, v_ref = tlognormal_moments_big(μ, σ, 1.0, side)
            # skip moments that are not representable in Float64
            (floatmin() < m_ref < floatmax() && floatmin() < v_ref < floatmax()) || continue
            d = side === :lower ? truncated(LogNormal(μ, σ); lower=1.0) : truncated(LogNormal(μ, σ); upper=1.0)
            @test mean(d) ≈ m_ref rtol=5e-14
            @test var(d) ≈ v_ref rtol=5e-14
        end
    end

    @testset "derivatives" begin
        for (μ, σ, x, side) in (
            (-3.0, 0.2, 1.0, :lower), (0.1, 0.5, 1.3, :lower), (-0.4, 0.2, 1.0, :lower), (2.0, 1.5, 0.5, :lower),
            (3.0, 0.2, 1.0, :upper), (0.1, 0.5, 1.3, :upper), (-2.0, 1.0, 1.0, :upper),
            # bounds far below (lower) or above (upper) the median, where erfcx of the standardized
            # quadrature points would overflow
            (40.0, 1.0, 1.0, :lower), (37.0, 0.2, 1.0, :lower), (-40.0, 1.0, 1.0, :upper),
        )
            θ = [μ, σ, x]
            tdist(θ) = side === :lower ? truncated(LogNormal(θ[1], θ[2]); lower=θ[3]) : truncated(LogNormal(θ[1], θ[2]); upper=θ[3])
            for (k, f) in ((1, mean), (2, var))
                g = ForwardDiff.gradient(θ -> f(tdist(θ)), θ)
                # central differences of the high-precision moments
                g_ref = map(1:3) do i
                    h = 1e-20 * max(abs(θ[i]), 1)
                    θp = big.(θ); θp[i] += h
                    θm = big.(θ); θm[i] -= h
                    (tlognormal_moments_big(θp..., side)[k] - tlognormal_moments_big(θm..., side)[k]) / 2h
                end
                @test g ≈ g_ref rtol=1e-12
            end
        end
    end

    @testset "bounds that do not truncate" begin
        d0 = LogNormal(0.3, 0.7)
        for l in (0, 0.0, -1.0, -Inf)
            @test mean(truncated(d0; lower=l)) == mean(d0)
            @test var(truncated(d0; lower=l)) == var(d0)
        end
        @test mean(truncated(d0; upper=Inf)) == mean(d0)
        @test var(truncated(d0; upper=Inf)) == var(d0)
        # an upper bound outside of the support leaves no mass
        for u in (0.0, -1.0)
            @test isnan(mean(truncated(d0; upper=u)))
            @test isnan(var(truncated(d0; upper=u)))
        end
    end

    @testset "degenerate log-normal distribution" begin
        @test mean(truncated(LogNormal(0.0, 0.0); lower=0.5)) == 1
        @test var(truncated(LogNormal(0.0, 0.0); lower=0.5)) == 0
        @test mean(truncated(LogNormal(0.0, 0.0); upper=2.0)) == 1
        @test var(truncated(LogNormal(0.0, 0.0); upper=2.0)) == 0
    end

    @testset "types" begin
        for T in (Float32, Float64)
            for d in (truncated(LogNormal(T(-3), T(0.2)); lower=T(1)), truncated(LogNormal(T(3), T(0.2)); upper=T(1)))
                m = @inferred mean(d)
                v = @inferred var(d)
                @test m isa T
                @test v isa T
                side = d.lower === nothing ? :upper : :lower
                m_ref, v_ref = tlognormal_moments_big(params(d.untruncated)..., something(d.lower, d.upper), side)
                @test m ≈ m_ref rtol=10eps(T)
                @test v ≈ v_ref rtol=10eps(T)
            end
        end
        # mixed parameter and bound types: the moments have the type of the truncation's mass
        for d in (truncated(LogNormal(-3.0f0, 0.2f0); lower=1), truncated(LogNormal(-3, 1); upper=1.0f0))
            @test @inferred(mean(d)) isa typeof(d.tp)
            @test @inferred(var(d)) isa typeof(d.tp)
        end
        # higher precision
        d = truncated(LogNormal(big(-3.0), big(1e-20)); lower=exp(big(-3.0) + big(2e-20)))
        m_ref, v_ref = tlognormal_moments_big(big(-3.0), big(1e-20), exp(big(-3.0) + big(2e-20)), :lower)
        @test mean(d) isa BigFloat
        @test mean(d) ≈ m_ref rtol=1e-50
        @test var(d) ≈ v_ref rtol=1e-30
    end
end
