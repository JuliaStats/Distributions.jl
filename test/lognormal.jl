using Distributions
using ForwardDiff
using Test

isnan_type(::Type{T}, v) where {T} = isnan(v) && v isa T

@testset "LogNormal" begin
    @test isa(convert(LogNormal{Float64}, Float16(0), Float16(1)),
              LogNormal{Float64})
    @test logpdf(LogNormal(0, 0), 1) === Inf
    @test logpdf(LogNormal(), Inf) === -Inf
    @test iszero(logcdf(LogNormal(0, 0), 1))
    @test iszero(logcdf(LogNormal(), Inf))
    @test logdiffcdf(LogNormal(), Float32(exp(5)), Float32(exp(3))) ≈ -6.607938594596893 rtol=1e-12
    @test logdiffcdf(LogNormal(), Float32(exp(5)), Float64(exp(3))) ≈ -6.60793859457367 rtol=1e-12
    @test logdiffcdf(LogNormal(), Float64(exp(5)), Float64(exp(3))) ≈ -6.607938594596893 rtol=1e-12
    let d = LogNormal(Float64(0), Float64(1)), x = Float64(exp(-60)), y = Float64(exp(-60.001))
        float_res = logdiffcdf(d, x, y)
        big_float_res = log(cdf(d, BigFloat(x, 100)) - cdf(d, BigFloat(y, 100)))
        @test float_res ≈ big_float_res
    end

    @test logccdf(LogNormal(0, 0), 1) === -Inf
    @test iszero(logccdf(LogNormal(eps(), 0), 1))

    @test iszero(quantile(LogNormal(), 0))
    @test isone(quantile(LogNormal(), 0.5))
    @test quantile(LogNormal(), 1) === Inf

    @test iszero(quantile(LogNormal(0, 0), 0))
    @test isone(quantile(LogNormal(0, 0), 0.75))
    @test quantile(LogNormal(0, 0), 1) === Inf

    @test iszero(quantile(LogNormal(0.25, 0), 0))
    @test quantile(LogNormal(0.25, 0), 0.95) == exp(0.25)
    @test quantile(LogNormal(0.25, 0), 1) === Inf

    @test cquantile(LogNormal(), 0) === Inf
    @test isone(cquantile(LogNormal(), 0.5))
    @test iszero(cquantile(LogNormal(), 1))

    @test cquantile(LogNormal(0, 0), 0) === Inf
    @test isone(cquantile(LogNormal(0, 0), 0.75))
    @test iszero(cquantile(LogNormal(0, 0), 1))

    @test cquantile(LogNormal(0.25, 0), 0) === Inf
    @test cquantile(LogNormal(0.25, 0), 0.95) == exp(0.25)
    @test iszero(cquantile(LogNormal(0.25, 0), 1))

    @test iszero(invlogcdf(LogNormal(), -Inf))
    @test isnan_type(Float64, invlogcdf(LogNormal(), NaN))

    @test invlogccdf(LogNormal(), -Inf) === Inf
    @test isnan_type(Float64, invlogccdf(LogNormal(), NaN))

    # test for #996 being fixed
    let d = LogNormal(0, 1), x = exp(1), ∂x = exp(2)
        @inferred cdf(d, ForwardDiff.Dual(x, ∂x)) ≈ ForwardDiff.Dual(cdf(d, x), ∂x * pdf(d, x))
    end
end


@testset "LogNormal type inference" begin
    # pdf
    @test @inferred(pdf(LogNormal(0.0, 0.0), 1.0)) === Inf
    @test @inferred(pdf(LogNormal(0.0, 0.0), 0.5)) === 0.0
    @test @inferred(pdf(LogNormal(0.0, 0.0), 0.0)) === 0.0
    @test @inferred(pdf(LogNormal(0.0, 0.0), -0.5)) === 0.0

    @test @inferred(pdf(LogNormal(0.0, 0.0), 1.0f0)) === Inf
    @test @inferred(pdf(LogNormal(0.0f0, 0.0f0), 1.0)) === Inf
    @test @inferred(pdf(LogNormal(0.0f0, 0.0f0), 1.0f0)) === Inf32

    @test isnan_type(Float64, @inferred(pdf(LogNormal(0.0, 0.0), NaN)))
    @test isnan_type(Float64, @inferred(pdf(LogNormal(NaN, 0.0), 1.0f0)))
    @test isnan_type(Float64, @inferred(pdf(LogNormal(NaN, 0.0), 0.0f0)))
    @test isnan_type(Float64, @inferred(pdf(LogNormal(NaN, 0.0), -1.0f0)))

    @test isnan_type(Float32, @inferred(pdf(LogNormal(NaN32, 0.0f0), 1.0f0)))
    @test isnan_type(Float32, @inferred(pdf(LogNormal(NaN32, 0.0f0), 0.0f0)))
    @test isnan_type(Float32, @inferred(pdf(LogNormal(NaN32, 0.0f0), -1.0f0)))

    @test @inferred(pdf(LogNormal(0 // 1, 0 // 1), 1 // 1)) === Inf
    @test @inferred(pdf(LogNormal(0 // 1, 0 // 1), 0 // 1)) === 0.0
    @test @inferred(pdf(LogNormal(0 // 1, 0 // 1), -1 // 1)) === 0.0
    @test isnan_type(Float64, @inferred(pdf(LogNormal(0 // 1, 0 // 1), NaN)))

    @test @inferred(pdf(LogNormal(0.0, 0.0), BigInt(1))) == big(Inf)
    @test @inferred(pdf(LogNormal(0.0, 0.0), BigInt(0))) == big(0.0)
    @test @inferred(pdf(LogNormal(0.0, 0.0), BigInt(-1))) == big(0.0)
    @test @inferred(pdf(LogNormal(0.0, 0.0), BigFloat(1))) == big(Inf)
    @test @inferred(pdf(LogNormal(0.0, 0.0), BigFloat(0))) == big(0.0)
    @test @inferred(pdf(LogNormal(0.0, 0.0), BigFloat(-1))) == big(0.0)
    @test isnan_type(BigFloat, @inferred(pdf(LogNormal(0.0, 0.0), BigFloat(NaN))))

    # logpdf
    @test @inferred(logpdf(LogNormal(0.0, 0.0), 1.0)) === Inf
    @test @inferred(logpdf(LogNormal(0.0, 0.0), 0.5)) === -Inf
    @test @inferred(logpdf(LogNormal(0.0, 0.0), 0.0)) === -Inf
    @test @inferred(logpdf(LogNormal(0.0, 0.0), -0.5)) === -Inf

    @test @inferred(logpdf(LogNormal(0.0, 0.0), 1.0f0)) === Inf
    @test @inferred(logpdf(LogNormal(0.0f0, 0.0f0), 1.0)) === Inf
    @test @inferred(logpdf(LogNormal(0.0f0, 0.0f0), 1.0f0)) === Inf32

    @test isnan_type(Float64, @inferred(logpdf(LogNormal(0.0, 0.0), NaN)))
    @test isnan_type(Float64, @inferred(logpdf(LogNormal(NaN, 0.0), 1.0f0)))
    @test isnan_type(Float64, @inferred(logpdf(LogNormal(NaN, 0.0), 0.0f0)))
    @test isnan_type(Float64, @inferred(logpdf(LogNormal(NaN, 0.0), -1.0f0)))

    @test isnan_type(Float32, @inferred(logpdf(LogNormal(NaN32, 0.0f0), 1.0f0)))
    @test isnan_type(Float32, @inferred(logpdf(LogNormal(NaN32, 0.0f0), 0.0f0)))
    @test isnan_type(Float32, @inferred(logpdf(LogNormal(NaN32, 0.0f0), -1.0f0)))

    @test @inferred(logpdf(LogNormal(0 // 1, 0 // 1), 1 // 1)) === Inf
    @test @inferred(logpdf(LogNormal(0 // 1, 0 // 1), 0 // 1)) === -Inf
    @test @inferred(logpdf(LogNormal(0 // 1, 0 // 1), -1 // 1)) === -Inf
    @test isnan_type(Float64, @inferred(logpdf(LogNormal(0 // 1, 0 // 1), NaN)))

    @test @inferred(logpdf(LogNormal(0.0, 0.0), BigInt(1))) == big(Inf)
    @test @inferred(logpdf(LogNormal(0.0, 0.0), BigInt(0))) == big(-Inf)
    @test @inferred(logpdf(LogNormal(0.0, 0.0), BigInt(-1))) == big(-Inf)
    @test @inferred(logpdf(LogNormal(0.0, 0.0), BigFloat(1))) == big(Inf)
    @test @inferred(logpdf(LogNormal(0.0, 0.0), BigFloat(0))) == big(-Inf)
    @test @inferred(logpdf(LogNormal(0.0, 0.0), BigFloat(-1))) == big(-Inf)
    @test isnan_type(BigFloat, @inferred(logpdf(LogNormal(0.0, 0.0), BigFloat(NaN))))

    # cdf
    @test @inferred(cdf(LogNormal(0.0, 0.0), 1.0)) === 1.0
    @test @inferred(cdf(LogNormal(0.0, 0.0), 0.5)) === 0.0
    @test @inferred(cdf(LogNormal(0.0, 0.0), 0.0)) === 0.0
    @test @inferred(cdf(LogNormal(0.0, 0.0), -0.5)) === 0.0

    @test @inferred(cdf(LogNormal(0.0, 0.0), 1.0f0)) === 1.0
    @test @inferred(cdf(LogNormal(0.0f0, 0.0f0), 1.0)) === 1.0
    @test @inferred(cdf(LogNormal(0.0f0, 0.0f0), 1.0f0)) === 1.0f0

    @test isnan_type(Float64, @inferred(cdf(LogNormal(0.0, 0.0), NaN)))
    @test isnan_type(Float64, @inferred(cdf(LogNormal(NaN, 0.0), 1.0f0)))
    @test isnan_type(Float64, @inferred(cdf(LogNormal(NaN, 0.0), 0.0f0)))
    @test isnan_type(Float64, @inferred(cdf(LogNormal(NaN, 0.0), -1.0f0)))

    @test isnan_type(Float32, @inferred(cdf(LogNormal(NaN32, 0.0f0), 1.0f0)))
    @test isnan_type(Float32, @inferred(cdf(LogNormal(NaN32, 0.0f0), 0.0f0)))
    @test isnan_type(Float32, @inferred(cdf(LogNormal(NaN32, 0.0f0), -1.0f0)))

    @test @inferred(cdf(LogNormal(0 // 1, 0 // 1), 1 // 1)) === 1.0
    @test @inferred(cdf(LogNormal(0 // 1, 0 // 1), 0 // 1)) === 0.0
    @test @inferred(cdf(LogNormal(0 // 1, 0 // 1), -1 // 1)) === 0.0
    @test isnan_type(Float64, @inferred(cdf(LogNormal(0 // 1, 0 // 1), NaN)))

    @test @inferred(cdf(LogNormal(0.0, 0.0), BigInt(1))) == big(1.0)
    @test @inferred(cdf(LogNormal(0.0, 0.0), BigInt(0))) == big(0.0)
    @test @inferred(cdf(LogNormal(0.0, 0.0), BigInt(-1))) == big(0.0)
    @test @inferred(cdf(LogNormal(0.0, 0.0), BigFloat(1))) == big(1.0)
    @test @inferred(cdf(LogNormal(0.0, 0.0), BigFloat(0))) == big(0.0)
    @test @inferred(cdf(LogNormal(0.0, 0.0), BigFloat(-1))) == big(0.0)
    @test isnan_type(BigFloat, @inferred(cdf(LogNormal(0.0, 0.0), BigFloat(NaN))))

    # logcdf
    @test @inferred(logcdf(LogNormal(0.0, 0.0), 1.0)) === -0.0
    @test @inferred(logcdf(LogNormal(0.0, 0.0), 0.5)) === -Inf
    @test @inferred(logcdf(LogNormal(0.0, 0.0), 0.0)) === -Inf
    @test @inferred(logcdf(LogNormal(0.0, 0.0), -0.5)) === -Inf

    @test @inferred(logcdf(LogNormal(0.0, 0.0), 1.0f0)) === -0.0
    @test @inferred(logcdf(LogNormal(0.0f0, 0.0f0), 1.0)) === -0.0
    @test @inferred(logcdf(LogNormal(0.0f0, 0.0f0), 1.0f0)) === -0.0f0

    @test isnan_type(Float64, @inferred(logcdf(LogNormal(0.0, 0.0), NaN)))
    @test isnan_type(Float64, @inferred(logcdf(LogNormal(NaN, 0.0), 1.0f0)))
    @test isnan_type(Float64, @inferred(logcdf(LogNormal(NaN, 0.0), 0.0f0)))
    @test isnan_type(Float64, @inferred(logcdf(LogNormal(NaN, 0.0), -1.0f0)))

    @test isnan_type(Float32, @inferred(logcdf(LogNormal(NaN32, 0.0f0), 1.0f0)))
    @test isnan_type(Float32, @inferred(logcdf(LogNormal(NaN32, 0.0f0), 0.0f0)))
    @test isnan_type(Float32, @inferred(logcdf(LogNormal(NaN32, 0.0f0), -1.0f0)))

    @test @inferred(logcdf(LogNormal(0 // 1, 0 // 1), 1 // 1)) === -0.0
    @test @inferred(logcdf(LogNormal(0 // 1, 0 // 1), 0 // 1)) === -Inf
    @test @inferred(logcdf(LogNormal(0 // 1, 0 // 1), -1 // 1)) === -Inf
    @test isnan_type(Float64, @inferred(logcdf(LogNormal(0 // 1, 0 // 1), NaN)))

    @test @inferred(logcdf(LogNormal(0.0, 0.0), BigInt(1))) == big(0.0)
    @test @inferred(logcdf(LogNormal(0.0, 0.0), BigInt(0))) == big(-Inf)
    @test @inferred(logcdf(LogNormal(0.0, 0.0), BigInt(-1))) == big(-Inf)
    @test @inferred(logcdf(LogNormal(0.0, 0.0), BigFloat(1))) == big(0.0)
    @test @inferred(logcdf(LogNormal(0.0, 0.0), BigFloat(0))) == big(-Inf)
    @test @inferred(logcdf(LogNormal(0.0, 0.0), BigFloat(-1))) == big(-Inf)
    @test isnan_type(BigFloat, @inferred(logcdf(LogNormal(0.0, 0.0), BigFloat(NaN))))

    # ccdf
    @test @inferred(ccdf(LogNormal(0.0, 0.0), 1.0)) === 0.0
    @test @inferred(ccdf(LogNormal(0.0, 0.0), 0.5)) === 1.0
    @test @inferred(ccdf(LogNormal(0.0, 0.0), 0.0)) === 1.0
    @test @inferred(ccdf(LogNormal(0.0, 0.0), -0.5)) === 1.0

    @test @inferred(ccdf(LogNormal(0.0, 0.0), 1.0f0)) === 0.0
    @test @inferred(ccdf(LogNormal(0.0f0, 0.0f0), 1.0)) === 0.0
    @test @inferred(ccdf(LogNormal(0.0f0, 0.0f0), 1.0f0)) === 0.0f0

    @test isnan_type(Float64, @inferred(ccdf(LogNormal(0.0, 0.0), NaN)))
    @test isnan_type(Float64, @inferred(ccdf(LogNormal(NaN, 0.0), 1.0f0)))
    @test isnan_type(Float64, @inferred(ccdf(LogNormal(NaN, 0.0), 0.0f0)))
    @test isnan_type(Float64, @inferred(ccdf(LogNormal(NaN, 0.0), -1.0f0)))

    @test isnan_type(Float32, @inferred(ccdf(LogNormal(NaN32, 0.0f0), 1.0f0)))
    @test isnan_type(Float32, @inferred(ccdf(LogNormal(NaN32, 0.0f0), 0.0f0)))
    @test isnan_type(Float32, @inferred(ccdf(LogNormal(NaN32, 0.0f0), -1.0f0)))

    @test @inferred(ccdf(LogNormal(0 // 1, 0 // 1), 1 // 1)) === 0.0
    @test @inferred(ccdf(LogNormal(0 // 1, 0 // 1), 0 // 1)) === 1.0
    @test @inferred(ccdf(LogNormal(0 // 1, 0 // 1), -1 // 1)) === 1.0
    @test isnan_type(Float64, @inferred(ccdf(LogNormal(0 // 1, 0 // 1), NaN)))

    @test @inferred(ccdf(LogNormal(0.0, 0.0), BigInt(1))) == big(0.0)
    @test @inferred(ccdf(LogNormal(0.0, 0.0), BigInt(0))) == big(1.0)
    @test @inferred(ccdf(LogNormal(0.0, 0.0), BigInt(-1))) == big(1.0)
    @test @inferred(ccdf(LogNormal(0.0, 0.0), BigFloat(1))) == big(0.0)
    @test @inferred(ccdf(LogNormal(0.0, 0.0), BigFloat(0))) == big(1.0)
    @test @inferred(ccdf(LogNormal(0.0, 0.0), BigFloat(-1))) == big(1.0)
    @test isnan_type(BigFloat, @inferred(ccdf(LogNormal(0.0, 0.0), BigFloat(NaN))))

    # logccdf
    @test @inferred(logccdf(LogNormal(0.0, 0.0), 1.0)) === -Inf
    @test @inferred(logccdf(LogNormal(0.0, 0.0), 0.5)) === -0.0
    @test @inferred(logccdf(LogNormal(0.0, 0.0), 0.0)) === -0.0
    @test @inferred(logccdf(LogNormal(0.0, 0.0), -0.5)) === -0.0

    @test @inferred(logccdf(LogNormal(0.0, 0.0), 1.0f0)) === -Inf
    @test @inferred(logccdf(LogNormal(0.0f0, 0.0f0), 1.0)) === -Inf
    @test @inferred(logccdf(LogNormal(0.0f0, 0.0f0), 1.0f0)) === -Inf32

    @test isnan_type(Float64, @inferred(logccdf(LogNormal(0.0, 0.0), NaN)))
    @test isnan_type(Float64, @inferred(logccdf(LogNormal(NaN, 0.0), 1.0f0)))
    @test isnan_type(Float64, @inferred(logccdf(LogNormal(NaN, 0.0), 0.0f0)))
    @test isnan_type(Float64, @inferred(logccdf(LogNormal(NaN, 0.0), -1.0f0)))

    @test isnan_type(Float32, @inferred(logccdf(LogNormal(NaN32, 0.0f0), 1.0f0)))
    @test isnan_type(Float32, @inferred(logccdf(LogNormal(NaN32, 0.0f0), 0.0f0)))
    @test isnan_type(Float32, @inferred(logccdf(LogNormal(NaN32, 0.0f0), -1.0f0)))

    @test @inferred(logccdf(LogNormal(0 // 1, 0 // 1), 1 // 1)) === -Inf
    @test @inferred(logccdf(LogNormal(0 // 1, 0 // 1), 0 // 1)) === -0.0
    @test @inferred(logccdf(LogNormal(0 // 1, 0 // 1), -1 // 1)) === -0.0
    @test isnan_type(Float64, @inferred(logccdf(LogNormal(0 // 1, 0 // 1), NaN)))

    @test @inferred(logccdf(LogNormal(0.0, 0.0), BigInt(1))) == big(-Inf)
    @test @inferred(logccdf(LogNormal(0.0, 0.0), BigInt(0))) == big(0.0)
    @test @inferred(logccdf(LogNormal(0.0, 0.0), BigInt(-1))) == big(0.0)
    @test @inferred(logccdf(LogNormal(0.0, 0.0), BigFloat(1))) == big(-Inf)
    @test @inferred(logccdf(LogNormal(0.0, 0.0), BigFloat(0))) == big(0.0)
    @test @inferred(logccdf(LogNormal(0.0, 0.0), BigFloat(-1))) == big(0.0)
    @test isnan_type(BigFloat, @inferred(logccdf(LogNormal(0.0, 0.0), BigFloat(NaN))))

    # quantile
    @test @inferred(quantile(LogNormal(1.0, 0.0), 0.0f0)) === 0.0
    @test @inferred(quantile(LogNormal(1.0, 0.0f0), 1.0)) === Inf
    @test @inferred(quantile(LogNormal(1.0f0, 0.0), 0.5)) ===  exp(1)
    @test isnan_type(Float64, @inferred(quantile(LogNormal(1.0f0, 0.0), NaN)))
    @test @inferred(quantile(LogNormal(1.0f0, 0.0f0), 0.0f0)) === 0.0f0
    @test @inferred(quantile(LogNormal(1.0f0, 0.0f0), 1.0f0)) === Inf32
    @test @inferred(quantile(LogNormal(1.0f0, 0.0f0), 0.5f0)) === exp(1.0f0)
    @test isnan_type(Float32, @inferred(quantile(LogNormal(1.0f0, 0.0f0), NaN32)))
    @test @inferred(quantile(LogNormal(1//1, 0//1), 1//2)) === exp(1)

    # cquantile
    @test @inferred(cquantile(LogNormal(1.0, 0.0), 0.0f0)) === Inf
    @test @inferred(cquantile(LogNormal(1.0, 0.0f0), 1.0)) === 0.0
    @test @inferred(cquantile(LogNormal(1.0f0, 0.0), 0.5)) === exp(1)
    @test isnan_type(Float64, @inferred(cquantile(LogNormal(1.0f0, 0.0), NaN)))
    @test @inferred(cquantile(LogNormal(1.0f0, 0.0f0), 0.0f0)) === Inf32
    @test @inferred(cquantile(LogNormal(1.0f0, 0.0f0), 1.0f0)) === 0.0f0
    @test @inferred(cquantile(LogNormal(1.0f0, 0.0f0), 0.5f0)) === exp(1.0f0)
    @test isnan_type(Float32, @inferred(cquantile(LogNormal(1.0f0, 0.0f0), NaN32)))
    @test @inferred(cquantile(LogNormal(1//1, 0//1), 1//2)) === exp(1)

    # gradlogpdf
    @test @inferred(gradlogpdf(LogNormal(0.0, 1.0), 1.0)) === -1.0
    @test @inferred(gradlogpdf(LogNormal(0.0, 1.0), exp(-1))) === 0.0
    @test @inferred(gradlogpdf(LogNormal(0.0, 1.0), 0.0)) === 0.0
    @test @inferred(gradlogpdf(LogNormal(0.0, 1.0), -0.5)) === 0.0

    @test @inferred(gradlogpdf(LogNormal(0.0, 1.0), 1.0f0)) === -1.0
    @test @inferred(gradlogpdf(LogNormal(0.0f0, 1.0f0), 1.0)) === -1.0
    @test @inferred(gradlogpdf(LogNormal(0.0f0, 1.0f0), 1.0f0)) === -1.0f0

    @test isnan_type(Float64, @inferred(logccdf(LogNormal(0.0, 1.0), NaN)))
    @test isnan_type(Float64, @inferred(logccdf(LogNormal(NaN, 1.0), 1.0f0)))
    @test isnan_type(Float64, @inferred(logccdf(LogNormal(NaN, 1.0), 0.0f0)))
    @test isnan_type(Float64, @inferred(logccdf(LogNormal(NaN, 1.0), -1.0f0)))

    @test isnan_type(Float32, @inferred(logccdf(LogNormal(NaN32, 1.0f0), 1.0f0)))
    @test isnan_type(Float32, @inferred(logccdf(LogNormal(NaN32, 1.0f0), 0.0f0)))
    @test isnan_type(Float32, @inferred(logccdf(LogNormal(NaN32, 1.0f0), -1.0f0)))

    @test @inferred(gradlogpdf(LogNormal(0 // 1, 1 // 1), 1 // 1)) === -1.0
    @test @inferred(gradlogpdf(LogNormal(0 // 1, 1 // 1), 0 // 1)) === 0.0
    @test @inferred(gradlogpdf(LogNormal(0 // 1, 1 // 1), -1 // 1)) === 0.0
    @test isnan_type(Float64, @inferred(gradlogpdf(LogNormal(0 // 1, 1 // 1), NaN)))

    @test @inferred(gradlogpdf(LogNormal(0.0, 1.0), BigInt(1))) == big(-1.0)
    @test @inferred(gradlogpdf(LogNormal(0.0, 1.0), BigInt(0))) == big(0.0)
    @test @inferred(gradlogpdf(LogNormal(0.0, 1.0), BigInt(-1))) == big(0.0)
    @test @inferred(gradlogpdf(LogNormal(0.0, 1.0), BigFloat(1))) == big(-1.0)
    @test @inferred(gradlogpdf(LogNormal(0.0, 1.0), BigFloat(0))) == big(0.0)
    @test @inferred(gradlogpdf(LogNormal(0.0, 1.0), BigFloat(-1))) == big(0.0)
    @test isnan_type(BigFloat, @inferred(gradlogpdf(LogNormal(0.0, 1.0), BigFloat(NaN))))
end

@testset "LogNormal fit to stats" begin
    @testset "fit moments" begin
        D = LogNormal(1,0.6)
        M = Moments(mean(D), var(D))
        Dfit = fit(LogNormal, M)
        @test D ≈ Dfit
        # handle not giving variance
        @test_throws Exception fit(LogNormal, Moments(3.2))
    end;
    @testset "fit two quantiles" begin
        qpl = @qp_m(3)
        qpu = @qp_u(5)
        d = fit(LogNormal, qpl, qpu);
        @test quantile.(d, [qpl.p, qpu.p]) ≈ [qpl.q, qpu.q]
        d = fit(LogNormal, qpu, qpl) # sort
        @test quantile.(d, [qpl.p, qpu.p]) ≈ [qpl.q, qpu.q]
    end;
    @testset "fit to quantilepoint and mean" begin
        d = LogNormal(1,1)
        m = log(mean(d))
        qp = @qp(quantile(d,0.95),0.95)
        dfit = fit_mean_quantile(LogNormal, mean(d), qp)
        @test dfit ≈ d
        dfit = fit(LogNormal, mean(d), qp, Val(:mean))
        @test dfit ≈ d
        # with lower quantile
        qp = @qp(quantile(d,0.05),0.05)
        dfit = fit_mean_quantile(LogNormal, mean(d), qp)
        @test dfit ≈ d
        # very close to mean can give very different results:
        qp = @qp(mean(d)-1e-4,0.95)
        dfit = fit_mean_quantile(LogNormal, mean(d), qp)
        @test mean(dfit) ≈ mean(d) && quantile(dfit, qp.p) ≈ qp.q
    end;
    @testset "fit to quantilepoint and mode" begin
        d = LogNormal(1,1)
        m = log(mode(d))
        qp = @qp(quantile(d,0.95),0.95)
        dfit = fit_mode_quantile(LogNormal, mode(d), qp)
        @test dfit ≈ d
        dfit = fit(LogNormal, mode(d), qp, Val(:mode))
        @test dfit ≈ d
        # with lower quantile
        qp = @qp(quantile(d,0.025),0.025)
        dfit = fit_mode_quantile(LogNormal, mode(d), qp)
        @test mode(dfit) ≈ mode(d) && quantile(dfit, qp.p) ≈ qp.q
    end;
    @testset "fit to quantilepoint and median" begin
        d = LogNormal(1,1)
        qp = @qp(quantile(d,0.95),0.95)
        dfit = fit(LogNormal, median(d), qp, Val(:median))
        @test dfit ≈ d
    end;
end;

