using StatsFuns

@testset "Rician" begin

    d1 = Rician(0.0, 10.0)
    @test d1 isa Rician{Float64}
    @test params(d1) == (0.0, 10.0)
    @test shape(d1) == 0.0
    @test scale(d1) == 200.0
    @test partype(d1) === Float64
    @test eltype(d1) === Float64
    @test rand(d1) isa Float64

    # Reference: WolframAlpha
    @test @inferred(mean(d1))::Float64 ≈ 5 * sqrt(2 * π)
    @test @inferred(std(d1))::Float64 ≈ sqrt(200 - 50 * π)
    @test @inferred(var(d1))::Float64 ≈ 200 - 50 * π
    @test @inferred(mode(d1))::Float64 ≈ 10
    ps = 0:0.1:1
    @test all(splat(isapprox), zip(quantile.(d1, ps), @. sqrt(-200 * log1p(-ps))))
    @test all(splat(isapprox), zip(cquantile.(d1, ps), @. sqrt(-200 * log(ps))))
    @test all(splat(isapprox), zip(invlogcdf.(d1, log.(ps)), @. sqrt(-200 * log1p(-ps))))
    @test all(splat(isapprox), zip(invlogccdf.(d1, log.(ps)), @. sqrt(-200 * log(ps))))
    xs = 0:0.5:30 # 99th percentile is approx 30
    @test all(splat(isapprox), zip(pdf.(d1, xs), map(x -> x / 100 * exp(-x^2/200), xs)))
    @test all(splat(isapprox), zip(logpdf.(d1, xs), map(x -> log(x / 100) - x^2/200, xs)))
    @test all(splat(isapprox), zip(cdf.(d1, xs), @. 1 - exp(-xs^2/200)))
    @test all(splat(isapprox), zip(logcdf.(d1, xs), @. log1mexp(-xs^2/200)))
    @test all(splat(isapprox), zip(ccdf.(d1, xs), @. exp(-xs^2/200)))
    @test all(splat(isapprox), zip(logccdf.(d1, xs), @. -xs^2/200))

    d2 = Rayleigh(10.0)
    @test mean(d1) ≈ mean(d2)
    @test var(d1) ≈ var(d2)
    @test mode(d1) ≈ mode(d2)
    @test median(d1) ≈ median(d2)
    @test Base.Fix1(quantile, d1).([0.25, 0.45, 0.60, 0.80, 0.90]) ≈ Base.Fix1(quantile, d2).([0.25, 0.45, 0.60, 0.80, 0.90])
    @test Base.Fix1(pdf, d1).(0.0:0.1:1.0) ≈ Base.Fix1(pdf, d2).(0.0:0.1:1.0)
    @test Base.Fix1(cdf, d1).(0.0:0.1:1.0) ≈ Base.Fix1(cdf, d2).(0.0:0.1:1.0)

    d1 = Rician(10.0, 10.0)
    @test median(d1) == quantile(d1, 0.5)
    x = Base.Fix1(quantile, d1).([0.25, 0.45, 0.60, 0.80, 0.90])
    @test all(Base.Fix1(cdf, d1).(x) .≈ [0.25, 0.45, 0.60, 0.80, 0.90])

    # Reference: WolframAlpha
    @test mean(d1) ≈ 15.4857246055
    @test std(d1) ≈ 7.75837182934
    @test var(d1) ≈ 60.1923334423
    @test median(d1) ≈ 14.7547909179
    @test all(xy -> isapprox(xy...; rtol=1e-5), zip(quantile.(d1, [0.1, 0.25, 0.5, 0.75, 0.9]), [5.86808, 9.62923, 14.7548, 20.5189, 26.0195]))

    d1 = Rician(5.0, 5.0)
    # Reference: WolframAlpha
    @test mean(d1) ≈ 7.7428623027557
    @test std(d1) ≈ 3.8791859146687
    @test var(d1) ≈ 15.0480833605606
    @test median(d1) ≈ 7.37739545894
    @test all(xy -> isapprox(xy...; rtol=1e-5), zip(quantile.(d1, [0.1, 0.25, 0.5, 0.75, 0.9]), [2.93404, 4.81462, 7.3774, 10.2595, 13.0097]))

    x = rand(d1, 100000)
    d1 = fit(Rician, x)
    @test d1 isa Rician{Float64}
    @test params(d1)[1] ≈ 5.0 atol=0.2
    @test params(d1)[2] ≈ 5.0 atol=0.2

    d1 = Rician(10.0f0, 10.0f0)
    @test d1 isa Rician{Float32}
    @test params(d1) == (10.0f0, 10.0f0)
    @test shape(d1) == 0.5f0
    @test scale(d1) == 300.0f0
    @test partype(d1) === Float32
    @test eltype(d1) === Float64
    @test rand(d1) isa Float64
    d1_f64 = Rician(10.0, 10.0)
    @test @inferred(mean(d1))::Float32 ≈ Float32(mean(d1_f64))
    @test @inferred(std(d1))::Float32 ≈ Float32(std(d1_f64))
    @test @inferred(var(d1))::Float32 ≈ Float32(var(d1_f64))
    @test @inferred(median(d1))::Float32 ≈ Float32(median(d1_f64))

    d1 = Rician()
    @test d1 isa Rician{Float64}
    @test params(d1) == (0.0, 1.0)

    @test pdf(d1, -Inf) == 0.0
    @test pdf(d1, -1) == 0.0
    @test pdf(d1, Inf) == 0.0
    @test isnan(pdf(d1, NaN))

    @test logpdf(d1, -Inf) == -Inf
    @test logpdf(d1, -1) == -Inf
    @test logpdf(d1, Inf) == -Inf
    @test isnan(logpdf(d1, NaN))

    @test cdf(d1, -Inf) == 0.0
    @test cdf(d1, -1) == 0.0
    @test cdf(d1, Inf) == 1.0
    @test isnan(cdf(d1, NaN))

    @test logcdf(d1, -Inf) == -Inf
    @test logcdf(d1, -1) == -Inf
    @test logcdf(d1, Inf) == 0.0
    @test isnan(logcdf(d1, NaN))

    @inferred pdf(d1, -Inf32)
    @inferred pdf(d1, -1)
    @inferred pdf(d1, 1.0)
    @inferred pdf(d1, 1.0f0)
    @inferred pdf(d1, 1)
    @inferred pdf(d1, 1//2)
    @inferred pdf(d1, Inf)

    @inferred logpdf(d1, -Inf32)
    @inferred logpdf(d1, -1)
    @inferred logpdf(d1, 1.0)
    @inferred logpdf(d1, 1.0f0)
    @inferred logpdf(d1, 1)
    @inferred logpdf(d1, 1//2)
    @inferred logpdf(d1, Inf)

    @inferred cdf(d1, -Inf32)
    @inferred cdf(d1, -1)
    @inferred cdf(d1, 1.0)
    @inferred cdf(d1, 1.0f0)
    @inferred cdf(d1, 1)
    @inferred cdf(d1, 1//2)
    @inferred cdf(d1, Inf)

    @inferred logcdf(d1, -Inf32)
    @inferred logcdf(d1, -1)
    @inferred logcdf(d1, 1.0)
    @inferred logcdf(d1, 1.0f0)
    @inferred logcdf(d1, 1)
    @inferred logcdf(d1, 1//2)
    @inferred logcdf(d1, Inf)

    # issue #2035
    # reference values computed with WolframAlpha
    d = Rician(1.0, 0.12)
    @test mean(d) ≈ 1.00722650704
    @test std(d) ≈ 0.119560710568
    @test var(d) ≈ 0.0142947635114
    @test median(d) ≈ 1.00719144719
    @test pdf(d, 0.5) ≈ 0.000400758853946
    @test pdf(d, 1) ≈ 3.33055235263
    @test pdf(d, 1.5) ≈ 0.000692437774661
    @test pdf(d, 2) ≈ 3.91711741719e-15
    @test logpdf(d, 0.5) ≈ -7.82215067328
    @test logpdf(d, 1) ≈ 1.2031381619
    @test logpdf(d, 1.5) ≈ -7.27529218003
    @test logpdf(d, 2) ≈ -33.1734203644
end
