@testset "Johnson" begin

    d1 = Johnson(0.0, 10.0, -2.0, 3.0)
    @test d1 isa Johnson{Float64}
    @test params(d1) == (0.0, 10.0, -2.0, 3.0)
    @test shape(d1) == 0.0
    @test scale(d1) == 10.0
    @test partype(d1) === Float64
    @test eltype(d1) === Float64
    @test rand(d1) isa Float64

    @test median(d1) == quantile(d1, 0.5)
    x = quantile.(d1, [0.25, 0.45, 0.60, 0.80, 0.90])
    @test all(cdf.(d1, x) .≈ [0.25, 0.45, 0.60, 0.80, 0.90])

    d1 = Johnson(10.0f0, 10.0f0, 1.0f0, 3.0f0)
    @test d1 isa Johnson{Float32}
    @test params(d1) == (10.0f0, 10.0f0, 1.0f0, 3.0f0)
    @test shape(d1) == 10.0f0
    @test scale(d1) == 10.0f0
    @test partype(d1) === Float32
    @test eltype(d1) === Float64
    @test rand(d1) isa Float64

    d1 = Johnson()
    @test d1 isa Johnson{Float64}
    @test params(d1) == (0.0, 1.0, 0.0, 1.0)

    @test pdf(d1, -Inf) == 0.0
    @test pdf(d1, Inf) == 0.0
    @test isnan(pdf(d1, NaN))

    @test logpdf(d1, -Inf) == -Inf
    @test logpdf(d1, Inf) == -Inf
    @test isnan(logpdf(d1, NaN))

    @test cdf(d1, -Inf) == 0.0
    @test cdf(d1, Inf) == 1.0
    @test isnan(cdf(d1, NaN))

    @test logcdf(d1, -Inf) == -Inf
    @test logcdf(d1, Inf) == 0.0
    @test isnan(logcdf(d1, NaN))

    @inferred pdf(d1, -Inf32)
    @inferred pdf(d1, 1.0)
    @inferred pdf(d1, 1.0f0)
    @inferred pdf(d1, 1)
    @inferred pdf(d1, 1//2)
    @inferred pdf(d1, Inf)

    @inferred logpdf(d1, -Inf32)
    @inferred logpdf(d1, 1.0)
    @inferred logpdf(d1, 1.0f0)
    @inferred logpdf(d1, 1)
    @inferred logpdf(d1, 1//2)
    @inferred logpdf(d1, Inf)

    @inferred cdf(d1, -Inf32)
    @inferred cdf(d1, 1.0)
    @inferred cdf(d1, 1.0f0)
    @inferred cdf(d1, 1)
    @inferred cdf(d1, 1//2)
    @inferred cdf(d1, Inf)

    @inferred logcdf(d1, -Inf32)
    @inferred logcdf(d1, 1.0)
    @inferred logcdf(d1, 1.0f0)
    @inferred logcdf(d1, 1)
    @inferred logcdf(d1, 1//2)
    @inferred logcdf(d1, Inf)

end
