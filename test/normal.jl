using Test, Distributions, ForwardDiff

isnan_type(::Type{T}, v) where {T} = isnan(v) && v isa T

@testset "Normal" begin
    @test isa(convert(Normal{Float64}, Float16(0), Float16(1)),
              Normal{Float64})
    @test Inf === logpdf(Normal(0, 0), 0)
    @test -Inf === logpdf(Normal(), Inf)
    @test iszero(logcdf(Normal(0, 0), 0))
    @test iszero(logcdf(Normal(), Inf))
    @test logdiffcdf(Normal(), Float32(5), Float32(3)) ≈ -6.6079385945968929 rtol=1e-12
    @test logdiffcdf(Normal(), Float64(5), Float64(3)) ≈ -6.6079385945968929 rtol=1e-12
    let d = Normal(Float64(0), Float64(1)), x = Float64(-60), y = Float64(-60.001)
        float_res = logdiffcdf(d, x, y)
        big_float_res = log(cdf(d, BigFloat(x, 100)) - cdf(d, BigFloat(y, 100)))
        @test float_res ≈ big_float_res
    end
    @test_throws ArgumentError logdiffcdf(Normal(), 1.0, 2.0)
    @test_throws MethodError logdiffcdf(Normal(), Float32(2), Float64(1))
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

    @test @inferred(cquantile(Normal(1.0, 0.0), 0.0f0))     ===  Inf
    @test @inferred(cquantile(Normal(1.0, 0.0f0), 1.0))     === -Inf
    @test @inferred(cquantile(Normal(1.0f0, 0.0), 0.5))     ===  1.0
    @test isnan_type(Float64, @inferred(cquantile(Normal(1.0f0, 0.0), NaN)))
    @test @inferred(cquantile(Normal(1.0f0, 0.0f0), 0.0f0)) ===  Inf32
    @test @inferred(cquantile(Normal(1.0f0, 0.0f0), 1.0f0)) === -Inf32
    @test @inferred(cquantile(Normal(1.0f0, 0.0f0), 0.5f0)) ===  1.0f0
    @test isnan_type(Float32, @inferred(cquantile(Normal(1.0f0, 0.0f0), NaN32)))
    @test @inferred(cquantile(Normal(1//1, 0//1), 1//2))    ===  1.0
end
