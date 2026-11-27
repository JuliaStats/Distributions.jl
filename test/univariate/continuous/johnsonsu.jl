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

    @testset "affine transformations" begin
        d = JohnsonSU(0.3, 1.5, -0.7, 2.0)
        ξ, λ, γ, δ = params(d)
        for (a, c) in ((2.0, 3.0), (2.0, -3.0))
            td = @inferred(a + c * d)
            @test td === JohnsonSU(a + c * ξ, abs(c) * λ, sign(c) * γ, δ)
            # `td` is the distribution of `a + c * X` with `X ~ d`
            for p in (0.01, 0.2, 0.5, 0.8, 0.99)
                x = quantile(d, p)
                @test cdf(td, a + c * x) ≈ (c > 0 ? cdf(d, x) : ccdf(d, x)) rtol = 1.0e-14
                @test logpdf(td, a + c * x) ≈ logpdf(d, x) - log(abs(c)) rtol = 1.0e-14
            end
        end

        # the remaining affine operations are defined in terms of `+` and `*`
        @test @inferred(2.0 + d) === JohnsonSU(ξ + 2.0, λ, γ, δ)
        @test @inferred(d - 2.0) === JohnsonSU(ξ - 2.0, λ, γ, δ)
        @test @inferred(d * 2.0) === JohnsonSU(2.0 * ξ, 2.0 * λ, γ, δ)
        @test @inferred(d / 2.0) === JohnsonSU(ξ / 2.0, λ / 2.0, γ, δ)
        @test @inferred(-d) === JohnsonSU(-ξ, λ, -γ, δ)

        # type promotion
        d32 = JohnsonSU(0.3f0, 1.5f0, -0.7f0, 2.0f0)
        @test @inferred(2 * d32 + 1) isa JohnsonSU{Float32}
        @test @inferred(2.0 * d32) isa JohnsonSU{Float64}

        # scaling by zero gives a degenerate distribution instead of an error
        d0 = @inferred(0.0 * d)
        @test d0 isa JohnsonSU{Float64}
        @test scale(d0) == 0
    end

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
