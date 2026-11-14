using StatsFuns: loghalf

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
end
