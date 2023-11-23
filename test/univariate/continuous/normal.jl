using Test, Distributions, StatsFuns, ForwardDiff, OffsetArrays

isnan_type(::Type{T}, v) where {T} = isnan(v) && v isa T

@testset "Normal" begin
    test_cgf(Normal(0,1  ), (1, -1, 100f0, 1e6, -1e6))
    test_cgf(Normal(1,0.4), (1, -1, 100f0, 1e6, -1e6))
    @test isa(convert(Normal{Float64}, Float16(0), Float16(1)),
              Normal{Float64})
    d = Normal(1.1, 2.3)
    @test convert(Normal{Float64}, d) === d
    d32 = convert(Normal{Float32}, d)
    @test d32 isa Normal{Float32}
    @test params(d32) == map(Float32, params(d))

    @test Inf === logpdf(Normal(0, 0), 0)
    @test -Inf === logpdf(Normal(), Inf)
    @test iszero(logcdf(Normal(0, 0), 0))
    @test iszero(logcdf(Normal(), Inf))
    @test @inferred(logdiffcdf(Normal(), 5f0, 3f0)) ≈ -6.607938594596893 rtol=1e-12
    @test @inferred(logdiffcdf(Normal(), 5f0, 3.0)) ≈ -6.607938594596893 rtol=1e-12
    @test @inferred(logdiffcdf(Normal(), 5.0, 3.0)) ≈ -6.607938594596893 rtol=1e-12
    @test_throws ArgumentError logdiffcdf(Normal(), 3, 5)

    # Arguments in the tails
    logdiffcdf_big(d::Normal, x::Real, y::Real) = logsubexp(logcdf(d, big(y)), logcdf(d, big(x)))
    for d in (Normal(), Normal(2.1, 0.1)), (a, b) in ((15, 10), (115, 100), (1015, 1000))
        for (x, y) in ((a, b), (-b, -a))
            @test isfinite(@inferred(logdiffcdf(d, x, y)))
            @test logdiffcdf(d, x, y) ≈ logdiffcdf_big(d, x, y)
        end
    end
    let d = Normal(Float64(0), Float64(1)), x = Float64(-60), y = Float64(-60.001)
        float_res = logdiffcdf(d, x, y)
        big_x = BigFloat(x; precision=100)
        big_y = BigFloat(y; precision=100)
        big_float_res = log(cdf(d, big_x) - cdf(d, big_y))
        @test float_res ≈ big_float_res
    end
    @test_throws ArgumentError logdiffcdf(Normal(), 1.0, 2.0)
    @test -Inf === logccdf(Normal(0, 0), 0)
    @test iszero(logccdf(Normal(eps(), 0), 0))
    @test -Inf === quantile(Normal(), 0)
    @test iszero(quantile(Normal(), 0.5))
    @test Inf === quantile(Normal(), 1)
    @test -Inf === quantile(Normal(0, 0), 0)
    @test iszero(quantile(Normal(0, 0), 0.75))
    @test Inf === quantile(Normal(0, 0), 1)
    @test -Inf === quantile(Normal(0.25, 0), 0)
    @test 0.25 == quantile(Normal(0.25, 0), 0.95)
    @test Inf === quantile(Normal(0.25, 0), 1)
    @test Inf === cquantile(Normal(), 0)
    @test iszero(cquantile(Normal(), 0.5))
    @test -Inf === cquantile(Normal(), 1)
    @test Inf === cquantile(Normal(0, 0), 0)
    @test iszero(cquantile(Normal(0, 0), 0.75))
    @test -Inf === cquantile(Normal(0, 0), 1)
    @test Inf === cquantile(Normal(0.25, 0), 0)
    @test 0.25 == cquantile(Normal(0.25, 0), 0.95)
    @test -Inf === cquantile(Normal(0.25, 0), 1)
    @test -Inf === invlogcdf(Normal(), -Inf)
    @test isnan_type(Float64, invlogcdf(Normal(), NaN))
    @test Inf === invlogccdf(Normal(), -Inf)
    @test isnan_type(Float64, invlogccdf(Normal(), NaN))
    # test for #996 being fixed
    let d = Normal(0, 1), x = 1.0, ∂x = 2.0
        @inferred cdf(d, ForwardDiff.Dual(x, ∂x)) ≈ ForwardDiff.Dual(cdf(d, x), ∂x * pdf(d, x))
    end
end

@testset "Normal logpdf & pdf type inference" begin
    @test @inferred(pdf(Normal(0.0, 0.0), 0.0))           === Inf
    @test @inferred(pdf(Normal(0.0, 0.0), -1.0))          === 0.0
    @test @inferred(pdf(Normal(0.0, 0.0), 0.0f0))         === Inf
    @test isnan_type(Float64, @inferred(pdf(Normal(0.0, 0.0), NaN)))
    @test @inferred(pdf(Normal(0.0f0, 0.0f0), 0.0))       === Inf
    @test @inferred(pdf(Normal(0.0f0, 0.0f0), 0.0f0))     === Inf32
    @test isnan_type(Float64, @inferred(pdf(Normal(0.0, 0.0), NaN)))
    @test isnan_type(Float64, @inferred(pdf(Normal(NaN, 0.0), 0.0f0)))
    @test isnan_type(Float32, @inferred(pdf(Normal(NaN32, 0.0f0), 0.0f0)))
    @test @inferred(pdf(Normal(0 // 1, 0 // 1), 0 // 1))  === Inf
    @test isnan_type(Float64, @inferred(pdf(Normal(0 // 1, 0 // 1), NaN)))
    @test @inferred(pdf(Normal(0.0, 0.0), BigInt(1)))     == big(0.0)
    @test @inferred(pdf(Normal(0.0, 0.0), BigFloat(1)))   == big(0.0)
    @test isnan_type(BigFloat, @inferred(pdf(Normal(0.0, 0.0), BigFloat(NaN))))

    @test @inferred(logpdf(Normal(0.0, 0.0), 0.0))           === Inf
    @test @inferred(logpdf(Normal(0.0, 0.0), -1.0))          === -Inf
    @test @inferred(logpdf(Normal(0.0, 0.0), 0.0f0))         === Inf
    @test isnan_type(Float64, @inferred(logpdf(Normal(0.0, 0.0), NaN)))
    @test @inferred(logpdf(Normal(0.0f0, 0.0f0), 0.0))       === Inf
    @test @inferred(logpdf(Normal(0.0f0, 0.0f0), 0.0f0))     === Inf32
    @test isnan_type(Float64, @inferred(logpdf(Normal(0.0, 0.0), NaN)))
    @test isnan_type(Float64, @inferred(logpdf(Normal(NaN, 0.0), 0.0f0)))
    @test isnan_type(Float32, @inferred(logpdf(Normal(NaN32, 0.0f0), 0.0f0)))
    @test @inferred(logpdf(Normal(0 // 1, 0 // 1), 0 // 1))  === Inf
    @test isnan_type(Float64, @inferred(logpdf(Normal(0 // 1, 0 // 1), NaN)))
    @test @inferred(logpdf(Normal(0.0, 0.0), BigInt(1)))     == big(-Inf)
    @test @inferred(logpdf(Normal(0.0, 0.0), BigFloat(1)))   == big(-Inf)
    @test isnan_type(BigFloat, @inferred(logpdf(Normal(0.0, 0.0), BigFloat(NaN))))

    @test @inferred(cdf(Normal(0.0, 0.0), 0.0))           === 1.0
    @test @inferred(cdf(Normal(0.0, 0.0), -1.0))          === 0.0
    @test @inferred(cdf(Normal(0.0, 0.0), 0.0f0))         === 1.0
    @test isnan_type(Float64, @inferred(cdf(Normal(0.0, 0.0), NaN)))
    @test @inferred(cdf(Normal(0.0f0, 0.0f0), 0.0))       === 1.0
    @test @inferred(cdf(Normal(0.0f0, 0.0f0), 0.0f0))     === 1.0f0
    @test isnan_type(Float64, @inferred(cdf(Normal(0.0, 0.0), NaN)))
    @test isnan_type(Float64, @inferred(cdf(Normal(NaN, 0.0), 0.0f0)))
    @test isnan_type(Float32, @inferred(cdf(Normal(NaN32, 0.0f0), 0.0f0)))
    @test @inferred(cdf(Normal(0 // 1, 0 // 1), 0 // 1))  === 1.0
    @test isnan_type(Float64, @inferred(cdf(Normal(0 // 1, 0 // 1), NaN)))
    @test @inferred(cdf(Normal(0.0, 0.0), BigInt(1)))     == big(1.0)
    @test @inferred(cdf(Normal(0.0, 0.0), BigFloat(1)))   == big(1.0)
    @test isnan_type(BigFloat, @inferred(cdf(Normal(0.0, 0.0), BigFloat(NaN))))

    @test @inferred(logcdf(Normal(0.0, 0.0), 0.0))           === -0.0
    @test @inferred(logcdf(Normal(0.0, 0.0), -1.0))          === -Inf
    @test @inferred(logcdf(Normal(0.0, 0.0), 0.0f0))         === -0.0
    @test isnan_type(Float64, @inferred(logcdf(Normal(0.0, 0.0), NaN)))
    @test @inferred(logcdf(Normal(0.0f0, 0.0f0), 0.0))       === -0.0
    @test @inferred(logcdf(Normal(0.0f0, 0.0f0), 0.0f0))     === -0.0f0
    @test isnan_type(Float64, @inferred(logcdf(Normal(0.0, 0.0), NaN)))
    @test isnan_type(Float64, @inferred(logcdf(Normal(NaN, 0.0), 0.0f0)))
    @test isnan_type(Float32, @inferred(logcdf(Normal(NaN32, 0.0f0), 0.0f0)))
    @test @inferred(logcdf(Normal(0 // 1, 0 // 1), 0 // 1))  === -0.0
    @test isnan_type(Float64, @inferred(logcdf(Normal(0 // 1, 0 // 1), NaN)))
    @test @inferred(logcdf(Normal(0.0, 0.0), BigInt(1)))     == big(0.0)
    @test @inferred(logcdf(Normal(0.0, 0.0), BigFloat(1)))   == big(0.0)
    @test isnan_type(BigFloat, @inferred(logcdf(Normal(0.0, 0.0), BigFloat(NaN))))

    @test @inferred(ccdf(Normal(0.0, 0.0), 0.0))           === 0.0
    @test @inferred(ccdf(Normal(0.0, 0.0), -1.0))          === 1.0
    @test @inferred(ccdf(Normal(0.0, 0.0), 0.0f0))         === 0.0
    @test isnan_type(Float64, @inferred(ccdf(Normal(0.0, 0.0), NaN)))
    @test @inferred(ccdf(Normal(0.0f0, 0.0f0), 0.0))       === 0.0
    @test @inferred(ccdf(Normal(0.0f0, 0.0f0), 0.0f0))     === 0.0f0
    @test isnan_type(Float64, @inferred(ccdf(Normal(0.0, 0.0), NaN)))
    @test isnan_type(Float64, @inferred(ccdf(Normal(NaN, 0.0), 0.0f0)))
    @test isnan_type(Float32, @inferred(ccdf(Normal(NaN32, 0.0f0), 0.0f0)))
    @test @inferred(ccdf(Normal(0 // 1, 0 // 1), 0 // 1))  === 0.0
    @test isnan_type(Float64, @inferred(ccdf(Normal(0 // 1, 0 // 1), NaN)))
    @test @inferred(ccdf(Normal(0.0, 0.0), BigInt(1)))     == big(0.0)
    @test @inferred(ccdf(Normal(0.0, 0.0), BigFloat(1)))   == big(0.0)
    @test isnan_type(BigFloat, @inferred(ccdf(Normal(0.0, 0.0), BigFloat(NaN))))

    @test @inferred(logccdf(Normal(0.0, 0.0), 0.0))           === -Inf
    @test @inferred(logccdf(Normal(0.0, 0.0), -1.0))          === -0.0
    @test @inferred(logccdf(Normal(0.0, 0.0), 0.0f0))         === -Inf
    @test isnan_type(Float64, @inferred(logccdf(Normal(0.0, 0.0), NaN)))
    @test @inferred(logccdf(Normal(0.0f0, 0.0f0), 0.0))       === -Inf
    @test @inferred(logccdf(Normal(0.0f0, 0.0f0), 0.0f0))     === -Inf32
    @test isnan_type(Float64, @inferred(logccdf(Normal(0.0, 0.0), NaN)))
    @test isnan_type(Float64, @inferred(logccdf(Normal(NaN, 0.0), 0.0f0)))
    @test isnan_type(Float32, @inferred(logccdf(Normal(NaN32, 0.0f0), 0.0f0)))
    @test @inferred(logccdf(Normal(0 // 1, 0 // 1), 0 // 1))  === -Inf
    @test isnan_type(Float64, @inferred(logccdf(Normal(0 // 1, 0 // 1), NaN)))
    @test @inferred(logccdf(Normal(0.0, 0.0), BigInt(1)))     == big(-Inf)
    @test @inferred(logccdf(Normal(0.0, 0.0), BigFloat(1)))   == big(-Inf)
    @test isnan_type(BigFloat, @inferred(logccdf(Normal(0.0, 0.0), BigFloat(NaN))))

    @test @inferred(quantile(Normal(1.0, 0.0), 0.0f0))     === -Inf
    @test @inferred(quantile(Normal(1.0, 0.0f0), 1.0))     ===  Inf
    @test @inferred(quantile(Normal(1.0f0, 0.0), 0.5))     ===  1.0
    @test isnan_type(Float64, @inferred(quantile(Normal(1.0f0, 0.0), NaN)))
    @test @inferred(quantile(Normal(1.0f0, 0.0f0), 0.0f0)) === -Inf32
    @test @inferred(quantile(Normal(1.0f0, 0.0f0), 1.0f0)) ===  Inf32
    @test @inferred(quantile(Normal(1.0f0, 0.0f0), 0.5f0)) ===  1.0f0
    @test isnan_type(Float32, @inferred(quantile(Normal(1.0f0, 0.0f0), NaN32)))
    @test @inferred(quantile(Normal(1//1, 0//1), 1//2))    ===  1.0
    @test @inferred(quantile(Normal(1f0, 0f0), 1//2))      ===  1f0
    @test @inferred(quantile(Normal(1f0, 0.0), 1//2))      ===  1.0

    @test @inferred(cquantile(Normal(1.0, 0.0), 0.0f0))     ===  Inf
    @test @inferred(cquantile(Normal(1.0, 0.0f0), 1.0))     === -Inf
    @test @inferred(cquantile(Normal(1.0f0, 0.0), 0.5))     ===  1.0
    @test isnan_type(Float64, @inferred(cquantile(Normal(1.0f0, 0.0), NaN)))
    @test @inferred(cquantile(Normal(1.0f0, 0.0f0), 0.0f0)) ===  Inf32
    @test @inferred(cquantile(Normal(1.0f0, 0.0f0), 1.0f0)) === -Inf32
    @test @inferred(cquantile(Normal(1.0f0, 0.0f0), 0.5f0)) ===  1.0f0
    @test isnan_type(Float32, @inferred(cquantile(Normal(1.0f0, 0.0f0), NaN32)))
    @test @inferred(cquantile(Normal(1//1, 0//1), 1//2))    ===  1.0
    @test @inferred(cquantile(Normal(1f0, 0f0), 1//2))      ===  1f0
    @test @inferred(cquantile(Normal(1f0, 0.0), 1//2))      ===  1.0
end

@testset "Normal: Sampling with integer-valued parameters" begin
    d = Normal{Int}(0, 1)
    @test rand(d) isa Float64
    @test rand(d, 10) isa Vector{Float64}
    @test rand(d, (3, 2)) isa Matrix{Float64}
end

@testset "NormalCanon and conversion" begin
    @test canonform(Normal()) == NormalCanon()
    @test meanform(NormalCanon()) == Normal()
    @test meanform(canonform(Normal(0.25, 0.7))) ≈ Normal(0.25, 0.7)
    @test convert(NormalCanon, convert(Normal, NormalCanon(0.3, 0.8))) ≈ NormalCanon(0.3, 0.8)
    @test mean(canonform(Normal(0.25, 0.7))) ≈ 0.25
    @test std(canonform(Normal(0.25, 0.7))) ≈ 0.7
end

# affine transformations
test_affine_transformations(Normal, randn(), randn()^2)
test_affine_transformations(NormalCanon, randn()^2, randn()^2)

@testset "Normal suffstats and OffsetArrays" begin
    a = rand(Normal(), 11)
    wa = 1.0:11.0

    resulta = @inferred(suffstats(Normal, a))

    resultwa = @inferred(suffstats(Normal, a, wa))

    b = OffsetArray(a, -5:5)
    wb = OffsetArray(wa, -5:5)

    resultb = @inferred(suffstats(Normal, b))
    @test resulta == resultb

    resultwb = @inferred(suffstats(Normal, b, wb))
    @test resultwa == resultwb

    @test_throws DimensionMismatch suffstats(Normal, b, wa)
end
