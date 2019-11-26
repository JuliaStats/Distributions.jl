using Test, Distributions, ForwardDiff

@testset "Normal" begin
    @test isa(convert(Normal{Float64}, Float16(0), Float16(1)),
              Normal{Float64})
    @test Inf === logpdf(Normal(0, 0), 0)
    @test -Inf === logpdf(Normal(), Inf)
    @test iszero(logcdf(Normal(0, 0), 0))
    @test iszero(logcdf(Normal(), Inf))
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
    @test isnan(invlogcdf(Normal(), NaN))
    @test Inf === invlogccdf(Normal(), -Inf)
    @test isnan(invlogccdf(Normal(), NaN))
    # test for #996 being fixed
    let d = Normal(0, 1), x = 1.0, ∂x = 2.0
        @inferred cdf(d, ForwardDiff.Dual(x, ∂x)) ≈ ForwardDiff.Dual(cdf(d, x), ∂x * pdf(d, x))
    end
end

@testset "Normal logpdf & pdf type inference" begin
    @test @inferred(pdf(Normal(0.0, 0.0), 0.0))           === Inf
    @test @inferred(pdf(Normal(0.0, 0.0), -1.0))          === 0.0
    @test @inferred(pdf(Normal(0.0, 0.0), 0.0f0))         === Inf
    @test @inferred(pdf(Normal(0.0, 0.0), NaN))           === NaN
    @test @inferred(pdf(Normal(0.0f0, 0.0f0), 0.0))       === Inf
    @test @inferred(pdf(Normal(0.0f0, 0.0f0), 0.0f0))     === Inf32
    @test @inferred(pdf(Normal(0.0, 0.0), NaN))           === NaN
    @test @inferred(pdf(Normal(NaN, 0.0), 0.0f0))         === NaN
    @test @inferred(pdf(Normal(NaN32, 0.0f0), 0.0f0))     === NaN32
    @test @inferred(pdf(Normal(0 // 1, 0 // 1), 0 // 1))  === Inf
    @test @inferred(pdf(Normal(0 // 1, 0 // 1), NaN))     === NaN
    @test @inferred(pdf(Normal(0.0, 0.0), BigInt(1)))     == big(0.0)
    @test @inferred(pdf(Normal(0.0, 0.0), BigFloat(1)))   == big(0.0)
    @test isequal(@inferred(pdf(Normal(0.0, 0.0), BigFloat(NaN))), big(NaN))

    @test @inferred(logpdf(Normal(0.0, 0.0), 0.0))           === Inf
    @test @inferred(logpdf(Normal(0.0, 0.0), -1.0))          === -Inf
    @test @inferred(logpdf(Normal(0.0, 0.0), 0.0f0))         === Inf
    @test @inferred(isnan(logpdf(Normal(0.0, 0.0), NaN)))
    @test @inferred(logpdf(Normal(0.0f0, 0.0f0), 0.0))       === Inf
    @test @inferred(logpdf(Normal(0.0f0, 0.0f0), 0.0f0))     === Inf32
    @test @inferred(logpdf(Normal(0.0, 0.0), NaN))           === NaN
    @test @inferred(logpdf(Normal(NaN, 0.0), 0.0f0))         === NaN
    @test @inferred(logpdf(Normal(NaN32, 0.0f0), 0.0f0))     === NaN32
    @test @inferred(logpdf(Normal(0 // 1, 0 // 1), 0 // 1))  === Inf
    @test @inferred(logpdf(Normal(0 // 1, 0 // 1), NaN))     === NaN
    @test @inferred(logpdf(Normal(0.0, 0.0), BigInt(1)))     == big(-Inf)
    @test @inferred(logpdf(Normal(0.0, 0.0), BigFloat(1)))   == big(-Inf)
    @test isequal(@inferred(logpdf(Normal(0.0, 0.0), BigFloat(NaN))), big(NaN))

    @test @inferred(cdf(Normal(0.0, 0.0), 0.0))           === 1.0
    @test @inferred(cdf(Normal(0.0, 0.0), -1.0))          === 0.0
    @test @inferred(cdf(Normal(0.0, 0.0), 0.0f0))         === 1.0
    @test @inferred(cdf(Normal(0.0, 0.0), NaN))           === NaN
    @test @inferred(cdf(Normal(0.0f0, 0.0f0), 0.0))       === 1.0
    @test @inferred(cdf(Normal(0.0f0, 0.0f0), 0.0f0))     === 1.0f0
    @test @inferred(cdf(Normal(0.0, 0.0), NaN))           === NaN
    @test @inferred(cdf(Normal(NaN, 0.0), 0.0f0))         === NaN
    @test @inferred(cdf(Normal(NaN32, 0.0f0), 0.0f0))     === NaN32
    @test @inferred(cdf(Normal(0 // 1, 0 // 1), 0 // 1))  === 1.0
    @test @inferred(cdf(Normal(0 // 1, 0 // 1), NaN))     === NaN
    @test @inferred(cdf(Normal(0.0, 0.0), BigInt(1)))     == big(1.0)
    @test @inferred(cdf(Normal(0.0, 0.0), BigFloat(1)))   == big(1.0)
    @test isequal(@inferred(cdf(Normal(0.0, 0.0), BigFloat(NaN))), big(NaN))

    @test @inferred(logcdf(Normal(0.0, 0.0), 0.0))           === -0.0
    @test @inferred(logcdf(Normal(0.0, 0.0), -1.0))          === -Inf
    @test @inferred(logcdf(Normal(0.0, 0.0), 0.0f0))         === -0.0
    @test @inferred(logcdf(Normal(0.0, 0.0), NaN))           === NaN
    @test @inferred(logcdf(Normal(0.0f0, 0.0f0), 0.0))       === -0.0
    @test @inferred(logcdf(Normal(0.0f0, 0.0f0), 0.0f0))     === -0.0f0
    @test @inferred(logcdf(Normal(0.0, 0.0), NaN))           === NaN
    @test @inferred(logcdf(Normal(NaN, 0.0), 0.0f0))         === NaN
    @test @inferred(logcdf(Normal(NaN32, 0.0f0), 0.0f0))     === NaN32
    @test @inferred(logcdf(Normal(0 // 1, 0 // 1), 0 // 1))  === -0.0
    @test @inferred(logcdf(Normal(0 // 1, 0 // 1), NaN))     === NaN
    @test @inferred(logcdf(Normal(0.0, 0.0), BigInt(1)))     == big(0.0)
    @test @inferred(logcdf(Normal(0.0, 0.0), BigFloat(1)))   == big(0.0)
    @test isequal(@inferred(logcdf(Normal(0.0, 0.0), BigFloat(NaN))), big(NaN))

    @test @inferred(ccdf(Normal(0.0, 0.0), 0.0))           === 0.0
    @test @inferred(ccdf(Normal(0.0, 0.0), -1.0))          === 1.0
    @test @inferred(ccdf(Normal(0.0, 0.0), 0.0f0))         === 0.0
    @test @inferred(ccdf(Normal(0.0, 0.0), NaN))           === NaN
    @test @inferred(ccdf(Normal(0.0f0, 0.0f0), 0.0))       === 0.0
    @test @inferred(ccdf(Normal(0.0f0, 0.0f0), 0.0f0))     === 0.0f0
    @test @inferred(ccdf(Normal(0.0, 0.0), NaN))           === NaN
    @test @inferred(ccdf(Normal(NaN, 0.0), 0.0f0))         === NaN
    @test @inferred(ccdf(Normal(NaN32, 0.0f0), 0.0f0))     === NaN32
    @test @inferred(ccdf(Normal(0 // 1, 0 // 1), 0 // 1))  === 0.0
    @test @inferred(ccdf(Normal(0 // 1, 0 // 1), NaN))     === NaN
    @test @inferred(ccdf(Normal(0.0, 0.0), BigInt(1)))     == big(0.0)
    @test @inferred(ccdf(Normal(0.0, 0.0), BigFloat(1)))   == big(0.0)
    @test isequal(@inferred(ccdf(Normal(0.0, 0.0), BigFloat(NaN))), big(NaN))

    @test @inferred(logccdf(Normal(0.0, 0.0), 0.0))           === -Inf
    @test @inferred(logccdf(Normal(0.0, 0.0), -1.0))          === -0.0
    @test @inferred(logccdf(Normal(0.0, 0.0), 0.0f0))         === -Inf
    @test @inferred(logccdf(Normal(0.0, 0.0), NaN))           === NaN
    @test @inferred(logccdf(Normal(0.0f0, 0.0f0), 0.0))       === -Inf
    @test @inferred(logccdf(Normal(0.0f0, 0.0f0), 0.0f0))     === -Inf32
    @test @inferred(logccdf(Normal(0.0, 0.0), NaN))           === NaN
    @test @inferred(logccdf(Normal(NaN, 0.0), 0.0f0))         === NaN
    @test @inferred(logccdf(Normal(NaN32, 0.0f0), 0.0f0))     === NaN32
    @test @inferred(logccdf(Normal(0 // 1, 0 // 1), 0 // 1))  === -Inf
    @test @inferred(logccdf(Normal(0 // 1, 0 // 1), NaN))     === NaN
    @test @inferred(logccdf(Normal(0.0, 0.0), BigInt(1)))     == big(-Inf)
    @test @inferred(logccdf(Normal(0.0, 0.0), BigFloat(1)))   == big(-Inf)
    @test isequal(@inferred(logccdf(Normal(0.0, 0.0), BigFloat(NaN))), big(NaN))

    @test @inferred(quantile(Normal(1.0, 0.0), 0.0f0))     === -Inf
    @test @inferred(quantile(Normal(1.0, 0.0f0), 1.0))     ===  Inf
    @test @inferred(quantile(Normal(1.0f0, 0.0), 0.5))     ===  1.0
    @test @inferred(quantile(Normal(1.0f0, 0.0), NaN))     ===  NaN
    @test @inferred(quantile(Normal(1.0f0, 0.0f0), 0.0f0)) === -Inf32
    @test @inferred(quantile(Normal(1.0f0, 0.0f0), 1.0f0)) ===  Inf32
    @test @inferred(quantile(Normal(1.0f0, 0.0f0), 0.5f0)) ===  1.0f0
    @test @inferred(quantile(Normal(1.0f0, 0.0f0), NaN32)) ===  NaN32
    @test @inferred(quantile(Normal(1//1, 0//1), 1//2))    ===  1.0

    @test @inferred(cquantile(Normal(1.0, 0.0), 0.0f0))     ===  Inf
    @test @inferred(cquantile(Normal(1.0, 0.0f0), 1.0))     === -Inf
    @test @inferred(cquantile(Normal(1.0f0, 0.0), 0.5))     ===  1.0
    @test @inferred(cquantile(Normal(1.0f0, 0.0), NaN))     ===  NaN
    @test @inferred(cquantile(Normal(1.0f0, 0.0f0), 0.0f0)) ===  Inf32
    @test @inferred(cquantile(Normal(1.0f0, 0.0f0), 1.0f0)) === -Inf32
    @test @inferred(cquantile(Normal(1.0f0, 0.0f0), 0.5f0)) ===  1.0f0
    @test @inferred(cquantile(Normal(1.0f0, 0.0f0), NaN32)) ===  NaN32
    @test @inferred(cquantile(Normal(1//1, 0//1), 1//2))    ===  1.0
end
