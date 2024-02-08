@testset "JohnsonSU" begin

    d1 = JohnsonSU(0.0, 10.0, -2.0, 3.0)
    @test d1 isa JohnsonSU{Float64}
    @test params(d1) == (0.0, 10.0, -2.0, 3.0)
    @test shape(d1) == 0.0
    @test scale(d1) == 10.0
    @test partype(d1) === Float64
    @test eltype(d1) === Float64
    @test rand(d1) isa Float64

    @test median(d1) == quantile(d1, 0.5)
    x = Base.Fix1(quantile, d1).([0.25, 0.45, 0.60, 0.80, 0.90])
    @test all(Base.Fix1(cdf, d1).(x) .≈ [0.25, 0.45, 0.60, 0.80, 0.90])
    y = Base.Fix1(cquantile, d1).([0.25, 0.45, 0.60, 0.80, 0.90])
    @test all(Base.Fix1(ccdf, d1).(y) .≈ [0.25, 0.45, 0.60, 0.80, 0.90])

    @test mean(d1) ≈ 7.581281
    @test var(d1) ≈ 19.1969485

    d1 = JohnsonSU(10.0f0, 10.0f0, 1.0f0, 3.0f0)
    @test d1 isa JohnsonSU{Float32}
    @test params(d1) == (10.0f0, 10.0f0, 1.0f0, 3.0f0)
    @test shape(d1) == 10.0f0
    @test scale(d1) == 10.0f0
    @test partype(d1) === Float32
    @test eltype(d1) === Float64
    @test rand(d1) isa Float64

    d1 = JohnsonSU(1.0, 1, 0, 1)
    @test Base.convert(JohnsonSU{Float64}, d1) === d1
    @test Base.convert(JohnsonSU{Int}, d1) isa JohnsonSU{Int}

    d1 = JohnsonSU()
    @test d1 isa JohnsonSU{Int}
    @test params(d1) == (0, 1, 0, 1)

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

    @test ccdf(d1, -Inf) == 1.0
    @test ccdf(d1, Inf) == 0.0
    @test isnan(ccdf(d1, NaN))

    @test logccdf(d1, -Inf) == 0.0
    @test logccdf(d1, Inf) == -Inf
    @test isnan(logccdf(d1, NaN))

    @test invlogcdf(d1, -Inf) == -Inf
    @test isnan(invlogcdf(d1, Inf))
    @test isnan(invlogcdf(d1, NaN))

    @test invlogccdf(d1, -Inf) == Inf
    @test isnan(invlogccdf(d1, Inf))
    @test isnan(invlogccdf(d1, NaN))

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

    @inferred ccdf(d1, -Inf32)
    @inferred ccdf(d1, 1.0)
    @inferred ccdf(d1, 1.0f0)
    @inferred ccdf(d1, 1)
    @inferred ccdf(d1, 1//2)
    @inferred ccdf(d1, Inf)

    @inferred logccdf(d1, -Inf32)
    @inferred logccdf(d1, 1.0)
    @inferred logccdf(d1, 1.0f0)
    @inferred logccdf(d1, 1)
    @inferred logccdf(d1, 1//2)
    @inferred logccdf(d1, Inf)

    @inferred invlogcdf(d1, -Inf32)
    @inferred invlogcdf(d1, 1.0)
    @inferred invlogcdf(d1, 1.0f0)
    @inferred invlogcdf(d1, 1)
    @inferred invlogcdf(d1, 1//2)
    @inferred invlogcdf(d1, Inf)

    @inferred invlogccdf(d1, -Inf32)
    @inferred invlogccdf(d1, 1.0)
    @inferred invlogccdf(d1, 1.0f0)
    @inferred invlogccdf(d1, 1)
    @inferred invlogccdf(d1, 1//2)
    @inferred invlogccdf(d1, Inf)

end
