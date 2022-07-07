using Distributions
using ForwardDiff
using Test

isnan_type(::Type{T}, v) where {T} = isnan(v) && v isa T

@testset "Log10Normal" begin
    @test isa(convert(Log10Normal{Float64}, Float16(0), Float16(1)),
              Log10Normal{Float64})
    d = Log10Normal(0, 1)
    @test convert(Log10Normal{Float64}, d) === d
    @test convert(Log10Normal{Float32}, d) isa Log10Normal{Float32}

    @test logpdf(Log10Normal(0, 0), 1) === Inf
    @test logpdf(Log10Normal(), Inf) === -Inf
    @test iszero(logcdf(Log10Normal(0, 0), 1))
    @test iszero(logcdf(Log10Normal(), Inf))
    @test logdiffcdf(Log10Normal(), Float32(exp10(3)), Float32(exp10(3))) === -Inf
    @test logdiffcdf(Log10Normal(), Float32(exp10(5)), Float32(exp10(3))) ≈ -6.607938594596893 rtol=1e-12
    @test logdiffcdf(Log10Normal(), Float32(exp10(5)), Float64(exp10(3))) ≈ -6.607938594596893 rtol=1e-12
    @test logdiffcdf(Log10Normal(), Float64(exp10(5)), Float64(exp10(3))) ≈ -6.607938594596893 rtol=1e-12
    let d = Log10Normal(Float64(0), Float64(1)), x = Float64(exp10(-60)), y = Float64(exp10(-60.001))
        float_res = logdiffcdf(d, x, y)
        big_x = BigFloat(x; precision=100)
        big_y = BigFloat(y; precision=100)
        big_float_res = log(cdf(d, big_x) - cdf(d, big_y))
        @test float_res ≈ big_float_res
    end

    @test logccdf(Log10Normal(0, 0), 1) === -Inf
    @test iszero(logccdf(Log10Normal(eps(), 0), 1))

    @test iszero(quantile(Log10Normal(), 0))
    @test isone(quantile(Log10Normal(), 0.5))
    @test quantile(Log10Normal(), 1) === Inf

    @test iszero(quantile(Log10Normal(0, 0), 0))
    @test isone(quantile(Log10Normal(0, 0), 0.75))
    @test quantile(Log10Normal(0, 0), 1) === Inf

    @test iszero(quantile(Log10Normal(0.25, 0), 0))
    @test quantile(Log10Normal(0.25, 0), 0.95) == exp10(0.25)
    @test quantile(Log10Normal(0.25, 0), 1) === Inf

    @test cquantile(Log10Normal(), 0) === Inf
    @test isone(cquantile(Log10Normal(), 0.5))
    @test iszero(cquantile(Log10Normal(), 1))

    @test cquantile(Log10Normal(0, 0), 0) === Inf
    @test isone(cquantile(Log10Normal(0, 0), 0.75))
    @test iszero(cquantile(Log10Normal(0, 0), 1))

    @test cquantile(Log10Normal(0.25, 0), 0) === Inf
    @test cquantile(Log10Normal(0.25, 0), 0.95) == exp10(0.25)
    @test iszero(cquantile(Log10Normal(0.25, 0), 1))

    @test iszero(invlogcdf(Log10Normal(), -Inf))
    @test isnan_type(Float64, invlogcdf(Log10Normal(), NaN))

    @test invlogccdf(Log10Normal(), -Inf) === Inf
    @test isnan_type(Float64, invlogccdf(Log10Normal(), NaN))

    # test for #996 being fixed
    let d = Log10Normal(0, 1), x = exp10(1), ∂x = exp10(2)
        @inferred cdf(d, ForwardDiff.Dual(x, ∂x)) ≈ ForwardDiff.Dual(cdf(d, x), ∂x * pdf(d, x))
    end
end


@testset "Log10Normal type inference" begin
    # pdf
    @test @inferred(pdf(Log10Normal(0.0, 0.0), 1.0)) === Inf
    @test @inferred(pdf(Log10Normal(0.0, 0.0), 0.5)) === 0.0
    @test @inferred(pdf(Log10Normal(0.0, 0.0), 0.0)) === 0.0
    @test @inferred(pdf(Log10Normal(0.0, 0.0), -0.5)) === 0.0

    @test @inferred(pdf(Log10Normal(0.0, 0.0), 1.0f0)) === Inf
    @test @inferred(pdf(Log10Normal(0.0f0, 0.0f0), 1.0)) === Inf
    @test @inferred(pdf(Log10Normal(0.0f0, 0.0f0), 1.0f0)) === Inf32

    @test isnan_type(Float64, @inferred(pdf(Log10Normal(0.0, 0.0), NaN)))
    @test isnan_type(Float64, @inferred(pdf(Log10Normal(NaN, 0.0), 1.0f0)))
    @test isnan_type(Float64, @inferred(pdf(Log10Normal(NaN, 0.0), 0.0f0)))
    @test isnan_type(Float64, @inferred(pdf(Log10Normal(NaN, 0.0), -1.0f0)))

    @test isnan_type(Float32, @inferred(pdf(Log10Normal(NaN32, 0.0f0), 1.0f0)))
    @test isnan_type(Float32, @inferred(pdf(Log10Normal(NaN32, 0.0f0), 0.0f0)))
    @test isnan_type(Float32, @inferred(pdf(Log10Normal(NaN32, 0.0f0), -1.0f0)))

    @test @inferred(pdf(Log10Normal(0 // 1, 0 // 1), 1 // 1)) === Inf
    @test @inferred(pdf(Log10Normal(0 // 1, 0 // 1), 0 // 1)) === 0.0
    @test @inferred(pdf(Log10Normal(0 // 1, 0 // 1), -1 // 1)) === 0.0
    @test isnan_type(Float64, @inferred(pdf(Log10Normal(0 // 1, 0 // 1), NaN)))

    @test @inferred(pdf(Log10Normal(0.0, 0.0), BigInt(1))) == big(Inf)
    @test @inferred(pdf(Log10Normal(0.0, 0.0), BigInt(0))) == big(0.0)
    @test @inferred(pdf(Log10Normal(0.0, 0.0), BigInt(-1))) == big(0.0)
    @test @inferred(pdf(Log10Normal(0.0, 0.0), BigFloat(1))) == big(Inf)
    @test @inferred(pdf(Log10Normal(0.0, 0.0), BigFloat(0))) == big(0.0)
    @test @inferred(pdf(Log10Normal(0.0, 0.0), BigFloat(-1))) == big(0.0)
    @test isnan_type(BigFloat, @inferred(pdf(Log10Normal(0.0, 0.0), BigFloat(NaN))))

    # logpdf
    @test @inferred(logpdf(Log10Normal(0.0, 0.0), 1.0)) === Inf
    @test @inferred(logpdf(Log10Normal(0.0, 0.0), 0.5)) === -Inf
    @test @inferred(logpdf(Log10Normal(0.0, 0.0), 0.0)) === -Inf
    @test @inferred(logpdf(Log10Normal(0.0, 0.0), -0.5)) === -Inf

    @test @inferred(logpdf(Log10Normal(0.0, 0.0), 1.0f0)) === Inf
    @test @inferred(logpdf(Log10Normal(0.0f0, 0.0f0), 1.0)) === Inf
    @test @inferred(logpdf(Log10Normal(0.0f0, 0.0f0), 1.0f0)) === Inf32

    @test isnan_type(Float64, @inferred(logpdf(Log10Normal(0.0, 0.0), NaN)))
    @test isnan_type(Float64, @inferred(logpdf(Log10Normal(NaN, 0.0), 1.0f0)))
    @test isnan_type(Float64, @inferred(logpdf(Log10Normal(NaN, 0.0), 0.0f0)))
    @test isnan_type(Float64, @inferred(logpdf(Log10Normal(NaN, 0.0), -1.0f0)))

    @test isnan_type(Float32, @inferred(logpdf(Log10Normal(NaN32, 0.0f0), 1.0f0)))
    @test isnan_type(Float32, @inferred(logpdf(Log10Normal(NaN32, 0.0f0), 0.0f0)))
    @test isnan_type(Float32, @inferred(logpdf(Log10Normal(NaN32, 0.0f0), -1.0f0)))

    @test @inferred(logpdf(Log10Normal(0 // 1, 0 // 1), 1 // 1)) === Inf
    @test @inferred(logpdf(Log10Normal(0 // 1, 0 // 1), 0 // 1)) === -Inf
    @test @inferred(logpdf(Log10Normal(0 // 1, 0 // 1), -1 // 1)) === -Inf
    @test isnan_type(Float64, @inferred(logpdf(Log10Normal(0 // 1, 0 // 1), NaN)))

    @test @inferred(logpdf(Log10Normal(0.0, 0.0), BigInt(1))) == big(Inf)
    @test @inferred(logpdf(Log10Normal(0.0, 0.0), BigInt(0))) == big(-Inf)
    @test @inferred(logpdf(Log10Normal(0.0, 0.0), BigInt(-1))) == big(-Inf)
    @test @inferred(logpdf(Log10Normal(0.0, 0.0), BigFloat(1))) == big(Inf)
    @test @inferred(logpdf(Log10Normal(0.0, 0.0), BigFloat(0))) == big(-Inf)
    @test @inferred(logpdf(Log10Normal(0.0, 0.0), BigFloat(-1))) == big(-Inf)
    @test isnan_type(BigFloat, @inferred(logpdf(Log10Normal(0.0, 0.0), BigFloat(NaN))))

    # cdf
    @test @inferred(cdf(Log10Normal(0.0, 0.0), 1.0)) === 1.0
    @test @inferred(cdf(Log10Normal(0.0, 0.0), 0.5)) === 0.0
    @test @inferred(cdf(Log10Normal(0.0, 0.0), 0.0)) === 0.0
    @test @inferred(cdf(Log10Normal(0.0, 0.0), -0.5)) === 0.0

    @test @inferred(cdf(Log10Normal(0.0, 0.0), 1.0f0)) === 1.0
    @test @inferred(cdf(Log10Normal(0.0f0, 0.0f0), 1.0)) === 1.0
    @test @inferred(cdf(Log10Normal(0.0f0, 0.0f0), 1.0f0)) === 1.0f0

    @test isnan_type(Float64, @inferred(cdf(Log10Normal(0.0, 0.0), NaN)))
    @test isnan_type(Float64, @inferred(cdf(Log10Normal(NaN, 0.0), 1.0f0)))
    @test isnan_type(Float64, @inferred(cdf(Log10Normal(NaN, 0.0), 0.0f0)))
    @test isnan_type(Float64, @inferred(cdf(Log10Normal(NaN, 0.0), -1.0f0)))

    @test isnan_type(Float32, @inferred(cdf(Log10Normal(NaN32, 0.0f0), 1.0f0)))
    @test isnan_type(Float32, @inferred(cdf(Log10Normal(NaN32, 0.0f0), 0.0f0)))
    @test isnan_type(Float32, @inferred(cdf(Log10Normal(NaN32, 0.0f0), -1.0f0)))

    @test @inferred(cdf(Log10Normal(0 // 1, 0 // 1), 1 // 1)) === 1.0
    @test @inferred(cdf(Log10Normal(0 // 1, 0 // 1), 0 // 1)) === 0.0
    @test @inferred(cdf(Log10Normal(0 // 1, 0 // 1), -1 // 1)) === 0.0
    @test isnan_type(Float64, @inferred(cdf(Log10Normal(0 // 1, 0 // 1), NaN)))

    @test @inferred(cdf(Log10Normal(0.0, 0.0), BigInt(1))) == big(1.0)
    @test @inferred(cdf(Log10Normal(0.0, 0.0), BigInt(0))) == big(0.0)
    @test @inferred(cdf(Log10Normal(0.0, 0.0), BigInt(-1))) == big(0.0)
    @test @inferred(cdf(Log10Normal(0.0, 0.0), BigFloat(1))) == big(1.0)
    @test @inferred(cdf(Log10Normal(0.0, 0.0), BigFloat(0))) == big(0.0)
    @test @inferred(cdf(Log10Normal(0.0, 0.0), BigFloat(-1))) == big(0.0)
    @test isnan_type(BigFloat, @inferred(cdf(Log10Normal(0.0, 0.0), BigFloat(NaN))))

    # logcdf
    @test @inferred(logcdf(Log10Normal(0.0, 0.0), 1.0)) === -0.0
    @test @inferred(logcdf(Log10Normal(0.0, 0.0), 0.5)) === -Inf
    @test @inferred(logcdf(Log10Normal(0.0, 0.0), 0.0)) === -Inf
    @test @inferred(logcdf(Log10Normal(0.0, 0.0), -0.5)) === -Inf

    @test @inferred(logcdf(Log10Normal(0.0, 0.0), 1.0f0)) === -0.0
    @test @inferred(logcdf(Log10Normal(0.0f0, 0.0f0), 1.0)) === -0.0
    @test @inferred(logcdf(Log10Normal(0.0f0, 0.0f0), 1.0f0)) === -0.0f0

    @test isnan_type(Float64, @inferred(logcdf(Log10Normal(0.0, 0.0), NaN)))
    @test isnan_type(Float64, @inferred(logcdf(Log10Normal(NaN, 0.0), 1.0f0)))
    @test isnan_type(Float64, @inferred(logcdf(Log10Normal(NaN, 0.0), 0.0f0)))
    @test isnan_type(Float64, @inferred(logcdf(Log10Normal(NaN, 0.0), -1.0f0)))

    @test isnan_type(Float32, @inferred(logcdf(Log10Normal(NaN32, 0.0f0), 1.0f0)))
    @test isnan_type(Float32, @inferred(logcdf(Log10Normal(NaN32, 0.0f0), 0.0f0)))
    @test isnan_type(Float32, @inferred(logcdf(Log10Normal(NaN32, 0.0f0), -1.0f0)))

    @test @inferred(logcdf(Log10Normal(0 // 1, 0 // 1), 1 // 1)) === -0.0
    @test @inferred(logcdf(Log10Normal(0 // 1, 0 // 1), 0 // 1)) === -Inf
    @test @inferred(logcdf(Log10Normal(0 // 1, 0 // 1), -1 // 1)) === -Inf
    @test isnan_type(Float64, @inferred(logcdf(Log10Normal(0 // 1, 0 // 1), NaN)))

    @test @inferred(logcdf(Log10Normal(0.0, 0.0), BigInt(1))) == big(0.0)
    @test @inferred(logcdf(Log10Normal(0.0, 0.0), BigInt(0))) == big(-Inf)
    @test @inferred(logcdf(Log10Normal(0.0, 0.0), BigInt(-1))) == big(-Inf)
    @test @inferred(logcdf(Log10Normal(0.0, 0.0), BigFloat(1))) == big(0.0)
    @test @inferred(logcdf(Log10Normal(0.0, 0.0), BigFloat(0))) == big(-Inf)
    @test @inferred(logcdf(Log10Normal(0.0, 0.0), BigFloat(-1))) == big(-Inf)
    @test isnan_type(BigFloat, @inferred(logcdf(Log10Normal(0.0, 0.0), BigFloat(NaN))))

    # ccdf
    @test @inferred(ccdf(Log10Normal(0.0, 0.0), 1.0)) === 0.0
    @test @inferred(ccdf(Log10Normal(0.0, 0.0), 0.5)) === 1.0
    @test @inferred(ccdf(Log10Normal(0.0, 0.0), 0.0)) === 1.0
    @test @inferred(ccdf(Log10Normal(0.0, 0.0), -0.5)) === 1.0

    @test @inferred(ccdf(Log10Normal(0.0, 0.0), 1.0f0)) === 0.0
    @test @inferred(ccdf(Log10Normal(0.0f0, 0.0f0), 1.0)) === 0.0
    @test @inferred(ccdf(Log10Normal(0.0f0, 0.0f0), 1.0f0)) === 0.0f0

    @test isnan_type(Float64, @inferred(ccdf(Log10Normal(0.0, 0.0), NaN)))
    @test isnan_type(Float64, @inferred(ccdf(Log10Normal(NaN, 0.0), 1.0f0)))
    @test isnan_type(Float64, @inferred(ccdf(Log10Normal(NaN, 0.0), 0.0f0)))
    @test isnan_type(Float64, @inferred(ccdf(Log10Normal(NaN, 0.0), -1.0f0)))

    @test isnan_type(Float32, @inferred(ccdf(Log10Normal(NaN32, 0.0f0), 1.0f0)))
    @test isnan_type(Float32, @inferred(ccdf(Log10Normal(NaN32, 0.0f0), 0.0f0)))
    @test isnan_type(Float32, @inferred(ccdf(Log10Normal(NaN32, 0.0f0), -1.0f0)))

    @test @inferred(ccdf(Log10Normal(0 // 1, 0 // 1), 1 // 1)) === 0.0
    @test @inferred(ccdf(Log10Normal(0 // 1, 0 // 1), 0 // 1)) === 1.0
    @test @inferred(ccdf(Log10Normal(0 // 1, 0 // 1), -1 // 1)) === 1.0
    @test isnan_type(Float64, @inferred(ccdf(Log10Normal(0 // 1, 0 // 1), NaN)))

    @test @inferred(ccdf(Log10Normal(0.0, 0.0), BigInt(1))) == big(0.0)
    @test @inferred(ccdf(Log10Normal(0.0, 0.0), BigInt(0))) == big(1.0)
    @test @inferred(ccdf(Log10Normal(0.0, 0.0), BigInt(-1))) == big(1.0)
    @test @inferred(ccdf(Log10Normal(0.0, 0.0), BigFloat(1))) == big(0.0)
    @test @inferred(ccdf(Log10Normal(0.0, 0.0), BigFloat(0))) == big(1.0)
    @test @inferred(ccdf(Log10Normal(0.0, 0.0), BigFloat(-1))) == big(1.0)
    @test isnan_type(BigFloat, @inferred(ccdf(Log10Normal(0.0, 0.0), BigFloat(NaN))))

    # logccdf
    @test @inferred(logccdf(Log10Normal(0.0, 0.0), 1.0)) === -Inf
    @test @inferred(logccdf(Log10Normal(0.0, 0.0), 0.5)) === -0.0
    @test @inferred(logccdf(Log10Normal(0.0, 0.0), 0.0)) === -0.0
    @test @inferred(logccdf(Log10Normal(0.0, 0.0), -0.5)) === -0.0

    @test @inferred(logccdf(Log10Normal(0.0, 0.0), 1.0f0)) === -Inf
    @test @inferred(logccdf(Log10Normal(0.0f0, 0.0f0), 1.0)) === -Inf
    @test @inferred(logccdf(Log10Normal(0.0f0, 0.0f0), 1.0f0)) === -Inf32

    @test isnan_type(Float64, @inferred(logccdf(Log10Normal(0.0, 0.0), NaN)))
    @test isnan_type(Float64, @inferred(logccdf(Log10Normal(NaN, 0.0), 1.0f0)))
    @test isnan_type(Float64, @inferred(logccdf(Log10Normal(NaN, 0.0), 0.0f0)))
    @test isnan_type(Float64, @inferred(logccdf(Log10Normal(NaN, 0.0), -1.0f0)))

    @test isnan_type(Float32, @inferred(logccdf(Log10Normal(NaN32, 0.0f0), 1.0f0)))
    @test isnan_type(Float32, @inferred(logccdf(Log10Normal(NaN32, 0.0f0), 0.0f0)))
    @test isnan_type(Float32, @inferred(logccdf(Log10Normal(NaN32, 0.0f0), -1.0f0)))

    @test @inferred(logccdf(Log10Normal(0 // 1, 0 // 1), 1 // 1)) === -Inf
    @test @inferred(logccdf(Log10Normal(0 // 1, 0 // 1), 0 // 1)) === -0.0
    @test @inferred(logccdf(Log10Normal(0 // 1, 0 // 1), -1 // 1)) === -0.0
    @test isnan_type(Float64, @inferred(logccdf(Log10Normal(0 // 1, 0 // 1), NaN)))

    @test @inferred(logccdf(Log10Normal(0.0, 0.0), BigInt(1))) == big(-Inf)
    @test @inferred(logccdf(Log10Normal(0.0, 0.0), BigInt(0))) == big(0.0)
    @test @inferred(logccdf(Log10Normal(0.0, 0.0), BigInt(-1))) == big(0.0)
    @test @inferred(logccdf(Log10Normal(0.0, 0.0), BigFloat(1))) == big(-Inf)
    @test @inferred(logccdf(Log10Normal(0.0, 0.0), BigFloat(0))) == big(0.0)
    @test @inferred(logccdf(Log10Normal(0.0, 0.0), BigFloat(-1))) == big(0.0)
    @test isnan_type(BigFloat, @inferred(logccdf(Log10Normal(0.0, 0.0), BigFloat(NaN))))

    # quantile
    @test @inferred(quantile(Log10Normal(1.0, 0.0), 0.0f0)) === 0.0
    @test @inferred(quantile(Log10Normal(1.0, 0.0f0), 1.0)) === Inf
    @test @inferred(quantile(Log10Normal(1.0f0, 0.0), 0.5)) ===  exp10(1)
    @test isnan_type(Float64, @inferred(quantile(Log10Normal(1.0f0, 0.0), NaN)))
    @test @inferred(quantile(Log10Normal(1.0f0, 0.0f0), 0.0f0)) === 0.0f0
    @test @inferred(quantile(Log10Normal(1.0f0, 0.0f0), 1.0f0)) === Inf32
    @test @inferred(quantile(Log10Normal(1.0f0, 0.0f0), 0.5f0)) === exp10(1.0f0)
    @test isnan_type(Float32, @inferred(quantile(Log10Normal(1.0f0, 0.0f0), NaN32)))
    @test @inferred(quantile(Log10Normal(1//1, 0//1), 1//2)) === exp10(1)

    # cquantile
    @test @inferred(cquantile(Log10Normal(1.0, 0.0), 0.0f0)) === Inf
    @test @inferred(cquantile(Log10Normal(1.0, 0.0f0), 1.0)) === 0.0
    @test @inferred(cquantile(Log10Normal(1.0f0, 0.0), 0.5)) === exp10(1)
    @test isnan_type(Float64, @inferred(cquantile(Log10Normal(1.0f0, 0.0), NaN)))
    @test @inferred(cquantile(Log10Normal(1.0f0, 0.0f0), 0.0f0)) === Inf32
    @test @inferred(cquantile(Log10Normal(1.0f0, 0.0f0), 1.0f0)) === 0.0f0
    @test @inferred(cquantile(Log10Normal(1.0f0, 0.0f0), 0.5f0)) === exp10(1.0f0)
    @test isnan_type(Float32, @inferred(cquantile(Log10Normal(1.0f0, 0.0f0), NaN32)))
    @test @inferred(cquantile(Log10Normal(1//1, 0//1), 1//2)) === exp10(1)

    # gradlogpdf
    @test @inferred(gradlogpdf(Log10Normal(0.0, 1.0), 1.0)) === -1.0
    # @test @inferred(gradlogpdf(Log10Normal(0.0, 1.0), 5^(-2*log(2)-log(5)) * exp(-log(2)^2))) ≈ 0.0 atol=1e-12
    @test @inferred(gradlogpdf(Log10Normal(0.0, 1.0), 0.0)) === 0.0
    @test @inferred(gradlogpdf(Log10Normal(0.0, 1.0), -0.5)) === 0.0

    @test @inferred(gradlogpdf(Log10Normal(0.0, 1.0), 1.0f0)) === -1.0
    @test @inferred(gradlogpdf(Log10Normal(0.0f0, 1.0f0), 1.0)) === -1.0
    @test @inferred(gradlogpdf(Log10Normal(0.0f0, 1.0f0), 1.0f0)) === -1.0f0

    @test isnan_type(Float64, @inferred(logccdf(Log10Normal(0.0, 1.0), NaN)))
    @test isnan_type(Float64, @inferred(logccdf(Log10Normal(NaN, 1.0), 1.0f0)))
    @test isnan_type(Float64, @inferred(logccdf(Log10Normal(NaN, 1.0), 0.0f0)))
    @test isnan_type(Float64, @inferred(logccdf(Log10Normal(NaN, 1.0), -1.0f0)))

    @test isnan_type(Float32, @inferred(logccdf(Log10Normal(NaN32, 1.0f0), 1.0f0)))
    @test isnan_type(Float32, @inferred(logccdf(Log10Normal(NaN32, 1.0f0), 0.0f0)))
    @test isnan_type(Float32, @inferred(logccdf(Log10Normal(NaN32, 1.0f0), -1.0f0)))

    @test @inferred(gradlogpdf(Log10Normal(0 // 1, 1 // 1), 1 // 1)) === -1.0
    @test @inferred(gradlogpdf(Log10Normal(0 // 1, 1 // 1), 0 // 1)) === 0.0
    @test @inferred(gradlogpdf(Log10Normal(0 // 1, 1 // 1), -1 // 1)) === 0.0
    @test isnan_type(Float64, @inferred(gradlogpdf(Log10Normal(0 // 1, 1 // 1), NaN)))

    @test @inferred(gradlogpdf(Log10Normal(0.0, 1.0), BigInt(1))) == big(-1.0)
    @test @inferred(gradlogpdf(Log10Normal(0.0, 1.0), BigInt(0))) == big(0.0)
    @test @inferred(gradlogpdf(Log10Normal(0.0, 1.0), BigInt(-1))) == big(0.0)
    @test @inferred(gradlogpdf(Log10Normal(0.0, 1.0), BigFloat(1))) == big(-1.0)
    @test @inferred(gradlogpdf(Log10Normal(0.0, 1.0), BigFloat(0))) == big(0.0)
    @test @inferred(gradlogpdf(Log10Normal(0.0, 1.0), BigFloat(-1))) == big(0.0)
    @test isnan_type(BigFloat, @inferred(gradlogpdf(Log10Normal(0.0, 1.0), BigFloat(NaN))))
end
