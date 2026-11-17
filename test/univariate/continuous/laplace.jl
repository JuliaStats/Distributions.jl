using StatsFuns: loghalf, logtwo

@testset "laplace.jl" begin
    # affine transformations
    test_cgf(Laplace(1, 1), (0.99, -0.99, 1f-2, -1f-5))
    test_affine_transformations(Laplace, randn(), randn()^2)

    @testset "logcdf and logccdf accuracy (T = $T)" for T in (Float32, Float64)
        d = Laplace(zero(T), one(T))
        # reference: log(1 - exp(-|z|) / 2) for z = x ≥ 0 (logcdf) or z = -x ≥ 0 (logccdf)
        ref(z) = T(setprecision(() -> log1p(-exp(-big(z)) / 2), BigFloat, 256))
        for z in T[0, 1.0e-10, 0.1, 1, 5, 20, 40, 100, 700]
            # `atol` only matters if the result is subnormal or underflows
            @test @inferred(logcdf(d, z)) ≈ ref(z) rtol = 4 * eps(T) atol = floatmin(T)
            @test @inferred(logccdf(d, -z)) ≈ ref(z) rtol = 4 * eps(T) atol = floatmin(T)
            # the other tail
            @test logcdf(d, -z) ≈ T(loghalf) - z rtol = 4 * eps(T)
            @test logccdf(d, z) ≈ T(loghalf) - z rtol = 4 * eps(T)
        end
    end

    @testset "invlogcdf and invlogccdf accuracy (T = $T)" for T in (Float16, Float32, Float64, BigFloat)
        d = Laplace(zero(T), one(T))
        # reference: log(2p) for p < 1/2 and -log(2(1 - p)) otherwise
        ref(lp) = T(
            setprecision(BigFloat, 2 * precision(T) + 64) do
                l = big(lp)
                2 * exp(l) < 1 ? logtwo + l : -log(-2 * expm1(l))
            end
        )
        m = T(loghalf)
        lps = [
            m, prevfloat(m), nextfloat(m), prevfloat(m, 16), nextfloat(m, 16), m - T(1 // 100), m + T(1 // 100),
            T(-logtwo / 2), T(-1 // 10), T(-1 // 1000), -one(T), -floatmin(T), T(-20), T(-200),
        ]
        for lp in lps
            @test @inferred(invlogcdf(d, lp)) ≈ ref(lp) rtol = 4 * eps(T)
            @test @inferred(invlogccdf(d, lp)) ≈ -ref(lp) rtol = 4 * eps(T)
        end
        @test invlogcdf(d, zero(T)) == T(Inf)
        @test invlogcdf(d, T(-Inf)) == T(-Inf)
        @test invlogccdf(d, zero(T)) == T(-Inf)
        @test invlogccdf(d, T(-Inf)) == T(Inf)
    end
end
