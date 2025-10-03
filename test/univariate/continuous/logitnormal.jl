# Tests on Univariate LogitNormal distributions

using Distributions
using Random, Test
using StatsFuns

####### Core testing procedure
function test_logitnormal(g::LogitNormal, n_tsamples::Int=10^6,
                          rng::Union{AbstractRNG, Missing} = missing)
    d = length(g)
    #mn = mean(g)
    md = median(g)
    #mo = mode(g)
    #s = var(g)
    #e = entropy(g)
    @test partype(g) == Float64
    #@test isa(mn, Float64)
    @test isa(md, Float64)
    #@test isa(mo, Float64)
    #@test isa(s, Float64)
    @test md ≈ logistic(g.μ)
    #@test entropy(g) ≈ d*(1 + Distributions.log2π)/2 + logdetcov(g.normal)/2 + sum(mean(g.normal))
     @test insupport(g,1e-8)
     # corner cases of 0 and 1 handled as in support
     @test insupport(g,1.0)
     @test pdf(g,0.0) == 0.0
     @test insupport(g,0.0)
     @test pdf(g,1.0) == 0.0
     @test !insupport(g,-1e-8)
     @test pdf(g,-1e-8) == 0.0
     @test !insupport(g,1+1e-8)
     @test pdf(g,1+1e-8) == 0.0

    # sampling
    if ismissing(rng)
        X = rand(g, n_tsamples)
    else
        X = rand(rng, g, n_tsamples)
    end
    @test isa(X, Array{Float64,1})

    # evaluation of logpdf and pdf
    for i = 1:min(100, n_tsamples)
        @test logpdf(g, X[i]) ≈ log(pdf(g, X[i]))
    end
    @test Base.Fix1(logpdf, g).(X) ≈ log.(Base.Fix1(pdf, g).(X))
    @test isequal(logpdf(g, 0),-Inf)
    @test isequal(logpdf(g, 1),-Inf)
    @test isequal(logpdf(g, -eps()),-Inf)

    # test the location and scale functions
    @test location(g) == g.μ
    @test scale(g) == g.σ
    @test params(g) == (g.μ, g.σ)
    @test g == deepcopy(g)
end

###### General Testing
@testset "Logitnormal tests" begin
    test_logitnormal( LogitNormal() )
    test_logitnormal( LogitNormal(2,0.5) )
    d = LogitNormal(Float32(2))
    typeof(rand(d, 5)) # still Float64
    @test convert(LogitNormal{Float32}, d) === d
    @test typeof(convert(LogitNormal{Float64}, d)) == typeof(LogitNormal(2,1))
end

@testset "LogitNormal: Degenerate case" begin
    # pdf
    @test @inferred(pdf(LogitNormal(0.0, 0.0), -1.0)) === 0.0
    @test @inferred(pdf(LogitNormal(0.0, 0.0), 0.0)) === 0.0
    @test @inferred(pdf(LogitNormal(0.0, 0.0), 0.5)) === Inf
    @test @inferred(pdf(LogitNormal(0.0, 0.0), 1.0)) === 0.0
    @test @inferred(pdf(LogitNormal(0.0, 0.0), 2.0)) === 0.0

    @test @inferred(pdf(LogitNormal(0.0, 0.0), 0.5f0)) === Inf
    @test @inferred(pdf(LogitNormal(0.0f0, 0.0f0), 0.5)) === Inf
    @test @inferred(pdf(LogitNormal(0.0f0, 0.0f0), 0.5f0)) === Inf32

    @test isnan(@inferred(pdf(LogitNormal(0.0, 0.0), NaN))::Float64)
    @test isnan(@inferred(pdf(LogitNormal(NaN, 0.0), -1.0f0))::Float64)
    @test isnan(@inferred(pdf(LogitNormal(NaN, 0.0), 0.0f0))::Float64)
    @test isnan(@inferred(pdf(LogitNormal(NaN, 0.0), 0.5f0))::Float64)
    @test isnan(@inferred(pdf(LogitNormal(NaN, 0.0), 1.0f0))::Float64)
    @test isnan(@inferred(pdf(LogitNormal(NaN, 0.0), 2.0f0))::Float64)

    @test isnan(@inferred(pdf(LogitNormal(0.0f0, 0.0f0), NaN32))::Float32)
    @test isnan(@inferred(pdf(LogitNormal(NaN32, 0.0f0), -1.0f0))::Float32)
    @test isnan(@inferred(pdf(LogitNormal(NaN32, 0.0f0), 0.0f0))::Float32)
    @test isnan(@inferred(pdf(LogitNormal(NaN32, 0.0f0), 0.5f0))::Float32)
    @test isnan(@inferred(pdf(LogitNormal(NaN32, 0.0f0), 1.0f0))::Float32)
    @test isnan(@inferred(pdf(LogitNormal(NaN32, 0.0f0), 2.0f0))::Float32)

    @test @inferred(pdf(LogitNormal(0 // 1, 0 // 1), -1 // 1)) === 0.0
    @test @inferred(pdf(LogitNormal(0 // 1, 0 // 1), 0 // 1)) === 0.0
    @test @inferred(pdf(LogitNormal(0 // 1, 0 // 1), 1 // 2)) === Inf
    @test @inferred(pdf(LogitNormal(0 // 1, 0 // 1), 1 // 1)) === 0.0
    @test @inferred(pdf(LogitNormal(0 // 1, 0 // 1), 2 // 1)) === 0.0
    @test isnan(@inferred(pdf(LogitNormal(0 // 1, 0 // 1), NaN))::Float64)

    @test @inferred(pdf(LogitNormal(0.0, 0.0), BigInt(-1))) == big(0.0)
    @test @inferred(pdf(LogitNormal(0.0, 0.0), BigInt(0))) == big(0.0)
    @test @inferred(pdf(LogitNormal(0.0, 0.0), BigInt(1)//BigInt(2))) == big(Inf)
    @test @inferred(pdf(LogitNormal(0.0, 0.0), BigInt(1))) == big(0.0)
    @test @inferred(pdf(LogitNormal(0.0, 0.0), BigInt(2))) == big(0.0)
    @test @inferred(pdf(LogitNormal(0.0, 0.0), BigFloat(-1))) == big(0.0)
    @test @inferred(pdf(LogitNormal(0.0, 0.0), BigFloat(0))) == big(0.0)
    @test @inferred(pdf(LogitNormal(0.0, 0.0), BigFloat(1//2))) == big(Inf)
    @test @inferred(pdf(LogitNormal(0.0, 0.0), BigFloat(1))) == big(0.0)
    @test @inferred(pdf(LogitNormal(0.0, 0.0), BigFloat(2))) == big(0.0)
    @test isnan(@inferred(pdf(LogitNormal(0.0, 0.0), BigFloat(NaN)))::BigFloat)

    # logpdf
    @test @inferred(logpdf(LogitNormal(0.0, 0.0), -1.0)) === -Inf
    @test @inferred(logpdf(LogitNormal(0.0, 0.0), 0.0)) === -Inf
    @test @inferred(logpdf(LogitNormal(0.0, 0.0), 0.5)) === Inf
    @test @inferred(logpdf(LogitNormal(0.0, 0.0), 1.0)) === -Inf
    @test @inferred(logpdf(LogitNormal(0.0, 0.0), 2.0)) === -Inf

    @test @inferred(logpdf(LogitNormal(0.0, 0.0), 0.5f0)) === Inf
    @test @inferred(logpdf(LogitNormal(0.0f0, 0.0f0), 0.5)) === Inf
    @test @inferred(logpdf(LogitNormal(0.0f0, 0.0f0), 0.5f0)) === Inf32

    @test isnan(@inferred(logpdf(LogitNormal(0.0, 0.0), NaN))::Float64)
    @test isnan(@inferred(logpdf(LogitNormal(NaN, 0.0), -1.0f0))::Float64)
    @test isnan(@inferred(logpdf(LogitNormal(NaN, 0.0), 0.0f0))::Float64)
    @test isnan(@inferred(logpdf(LogitNormal(NaN, 0.0), 0.5f0))::Float64)
    @test isnan(@inferred(logpdf(LogitNormal(NaN, 0.0), 1.0f0))::Float64)
    @test isnan(@inferred(logpdf(LogitNormal(NaN, 0.0), 2.0f0))::Float64)

    @test isnan(@inferred(logpdf(LogitNormal(NaN32, 0.0f0), -1.0f0))::Float32)
    @test isnan(@inferred(logpdf(LogitNormal(NaN32, 0.0f0), 0.0f0))::Float32)
    @test isnan(@inferred(logpdf(LogitNormal(NaN32, 0.0f0), 0.5f0))::Float32)
    @test isnan(@inferred(logpdf(LogitNormal(NaN32, 0.0f0), 1.0f0))::Float32)
    @test isnan(@inferred(logpdf(LogitNormal(NaN32, 0.0f0), 2.0f0))::Float32)

    @test @inferred(logpdf(LogitNormal(0 // 1, 0 // 1), -1 // 1)) === -Inf
    @test @inferred(logpdf(LogitNormal(0 // 1, 0 // 1), 0 // 1)) === -Inf
    @test @inferred(logpdf(LogitNormal(0 // 1, 0 // 1), 1 // 2)) === Inf
    @test @inferred(logpdf(LogitNormal(0 // 1, 0 // 1), 1 // 1)) === -Inf
    @test @inferred(logpdf(LogitNormal(0 // 1, 0 // 1), 2 // 1)) === -Inf
    @test isnan(@inferred(logpdf(LogitNormal(0 // 1, 0 // 1), NaN))::Float64)

    @test @inferred(logpdf(LogitNormal(0.0, 0.0), BigInt(-1))) == big(-Inf)
    @test @inferred(logpdf(LogitNormal(0.0, 0.0), BigInt(0))) == big(-Inf)
    @test @inferred(logpdf(LogitNormal(0.0, 0.0), BigInt(1)//BigInt(2))) == big(Inf)
    @test @inferred(logpdf(LogitNormal(0.0, 0.0), BigInt(1))) == big(-Inf)
    @test @inferred(logpdf(LogitNormal(0.0, 0.0), BigInt(2))) == big(-Inf)
    @test @inferred(logpdf(LogitNormal(0.0, 0.0), BigFloat(-1))) == big(-Inf)
    @test @inferred(logpdf(LogitNormal(0.0, 0.0), BigFloat(0))) == big(-Inf)
    @test @inferred(logpdf(LogitNormal(0.0, 0.0), BigFloat(1//2))) == big(Inf)
    @test @inferred(logpdf(LogitNormal(0.0, 0.0), BigFloat(1))) == big(-Inf)
    @test @inferred(logpdf(LogitNormal(0.0, 0.0), BigFloat(2))) == big(-Inf)
    @test isnan(@inferred(logpdf(LogitNormal(0.0, 0.0), BigFloat(NaN)))::BigFloat)

    # cdf
    @test @inferred(cdf(LogitNormal(0.0, 0.0), -1.0)) === 0.0
    @test @inferred(cdf(LogitNormal(0.0, 0.0), 0.0)) === 0.0
    @test @inferred(cdf(LogitNormal(0.0, 0.0), 0.5)) === 1.0
    @test @inferred(cdf(LogitNormal(0.0, 0.0), 1.0)) === 1.0
    @test @inferred(cdf(LogitNormal(0.0, 0.0), 2.0)) === 1.0

    @test @inferred(cdf(LogitNormal(0.0, 0.0), 0.5f0)) === 1.0
    @test @inferred(cdf(LogitNormal(0.0f0, 0.0f0), 0.5)) === 1.0
    @test @inferred(cdf(LogitNormal(0.0f0, 0.0f0), 0.5f0)) === 1.0f0

    @test isnan(@inferred(cdf(LogitNormal(0.0, 0.0), NaN))::Float64)
    @test isnan(@inferred(cdf(LogitNormal(NaN, 0.0), -1.0f0))::Float64)
    @test isnan(@inferred(cdf(LogitNormal(NaN, 0.0), 0.0f0))::Float64)
    @test isnan(@inferred(cdf(LogitNormal(NaN, 0.0), 0.5f0))::Float64)
    @test isnan(@inferred(cdf(LogitNormal(NaN, 0.0), 1.0f0))::Float64)
    @test isnan(@inferred(cdf(LogitNormal(NaN, 0.0), 2.0f0))::Float64)

    @test isnan(@inferred(cdf(LogitNormal(NaN32, 0.0f0), -1.0f0))::Float32)
    @test isnan(@inferred(cdf(LogitNormal(NaN32, 0.0f0), 0.0f0))::Float32)
    @test isnan(@inferred(cdf(LogitNormal(NaN32, 0.0f0), 0.5f0))::Float32)
    @test isnan(@inferred(cdf(LogitNormal(NaN32, 0.0f0), 1.0f0))::Float32)
    @test isnan(@inferred(cdf(LogitNormal(NaN32, 0.0f0), 2.0f0))::Float32)

    @test @inferred(cdf(LogitNormal(0 // 1, 0 // 1), -1 // 1)) === 0.0
    @test @inferred(cdf(LogitNormal(0 // 1, 0 // 1), 0 // 1)) === 0.0
    @test @inferred(cdf(LogitNormal(0 // 1, 0 // 1), 1 // 2)) === 1.0
    @test @inferred(cdf(LogitNormal(0 // 1, 0 // 1), 1 // 1)) === 1.0
    @test @inferred(cdf(LogitNormal(0 // 1, 0 // 1), 2 // 1)) === 1.0
    @test isnan(@inferred(cdf(LogitNormal(0 // 1, 0 // 1), NaN))::Float64)

    @test @inferred(cdf(LogitNormal(0.0, 0.0), BigInt(-1))) == big(0.0)
    @test @inferred(cdf(LogitNormal(0.0, 0.0), BigInt(0))) == big(0.0)
    @test @inferred(cdf(LogitNormal(0.0, 0.0), BigInt(1)//BigInt(2))) == big(1.0)
    @test @inferred(cdf(LogitNormal(0.0, 0.0), BigInt(1))) == big(1.0)
    @test @inferred(cdf(LogitNormal(0.0, 0.0), BigInt(2))) == big(1.0)
    @test @inferred(cdf(LogitNormal(0.0, 0.0), BigFloat(-1))) == big(0.0)
    @test @inferred(cdf(LogitNormal(0.0, 0.0), BigFloat(0))) == big(0.0)
    @test @inferred(cdf(LogitNormal(0.0, 0.0), BigFloat(1//2))) == big(1.0)
    @test @inferred(cdf(LogitNormal(0.0, 0.0), BigFloat(1))) == big(1.0)
    @test @inferred(cdf(LogitNormal(0.0, 0.0), BigFloat(2))) == big(1.0)
    @test isnan(@inferred(cdf(LogitNormal(0.0, 0.0), big(NaN)))::BigFloat)

    # logcdf
    @test @inferred(logcdf(LogitNormal(0.0, 0.0), -1.0)) === -Inf
    @test @inferred(logcdf(LogitNormal(0.0, 0.0), 0.0)) === -Inf
    @test @inferred(logcdf(LogitNormal(0.0, 0.0), 0.5)) === -0.0
    @test @inferred(logcdf(LogitNormal(0.0, 0.0), 1.0)) === -0.0
    @test @inferred(logcdf(LogitNormal(0.0, 0.0), 2.0)) === -0.0

    @test @inferred(logcdf(LogitNormal(0.0, 0.0), 0.5f0)) === -0.0
    @test @inferred(logcdf(LogitNormal(0.0f0, 0.0f0), 0.5)) === -0.0
    @test @inferred(logcdf(LogitNormal(0.0f0, 0.0f0), 0.5f0)) === -0.0f0

    @test isnan(@inferred(logcdf(LogitNormal(0.0, 0.0), NaN))::Float64)
    @test isnan(@inferred(logcdf(LogitNormal(NaN, 0.0), -1.0f0))::Float64)
    @test isnan(@inferred(logcdf(LogitNormal(NaN, 0.0), 0.0f0))::Float64)
    @test isnan(@inferred(logcdf(LogitNormal(NaN, 0.0), 0.5f0))::Float64)
    @test isnan(@inferred(logcdf(LogitNormal(NaN, 0.0), 1.0f0))::Float64)
    @test isnan(@inferred(logcdf(LogitNormal(NaN, 0.0), 2.0f0))::Float64)

    @test isnan(@inferred(logcdf(LogitNormal(NaN32, 0.0f0), -1.0f0))::Float32)
    @test isnan(@inferred(logcdf(LogitNormal(NaN32, 0.0f0), 0.0f0))::Float32)
    @test isnan(@inferred(logcdf(LogitNormal(NaN32, 0.0f0), 0.5f0))::Float32)
    @test isnan(@inferred(logcdf(LogitNormal(NaN32, 0.0f0), 1.0f0))::Float32)
    @test isnan(@inferred(logcdf(LogitNormal(NaN32, 0.0f0), 2.0f0))::Float32)

    @test @inferred(logcdf(LogitNormal(0 // 1, 0 // 1), -1 // 1)) === -Inf
    @test @inferred(logcdf(LogitNormal(0 // 1, 0 // 1), 0 // 1)) === -Inf
    @test @inferred(logcdf(LogitNormal(0 // 1, 0 // 1), 1 // 2)) === -0.0
    @test @inferred(logcdf(LogitNormal(0 // 1, 0 // 1), 1 // 1)) === -0.0
    @test @inferred(logcdf(LogitNormal(0 // 1, 0 // 1), 2 // 1)) === -0.0
    @test isnan(@inferred(logcdf(LogitNormal(0 // 1, 0 // 1), NaN))::Float64)

    @test @inferred(logcdf(LogitNormal(0.0, 0.0), BigInt(-1))) == big(-Inf)
    @test @inferred(logcdf(LogitNormal(0.0, 0.0), BigInt(0))) == big(-Inf)
    @test @inferred(logcdf(LogitNormal(0.0, 0.0), BigInt(1)//BigInt(2))) == big(0.0)
    @test @inferred(logcdf(LogitNormal(0.0, 0.0), BigInt(1))) == big(0.0)
    @test @inferred(logcdf(LogitNormal(0.0, 0.0), BigInt(2))) == big(0.0)
    @test @inferred(logcdf(LogitNormal(0.0, 0.0), BigFloat(-1))) == big(-Inf)
    @test @inferred(logcdf(LogitNormal(0.0, 0.0), BigFloat(0))) == big(-Inf)
    @test @inferred(logcdf(LogitNormal(0.0, 0.0), BigFloat(1//2))) == big(0.0)
    @test @inferred(logcdf(LogitNormal(0.0, 0.0), BigFloat(1))) == big(0.0)
    @test @inferred(logcdf(LogitNormal(0.0, 0.0), BigFloat(2))) == big(0.0)
    @test isnan(@inferred(logcdf(LogitNormal(0.0, 0.0), big(NaN)))::BigFloat)

    # ccdf
    @test @inferred(ccdf(LogitNormal(0.0, 0.0), -1.0)) === 1.0
    @test @inferred(ccdf(LogitNormal(0.0, 0.0), 0.0)) === 1.0
    @test @inferred(ccdf(LogitNormal(0.0, 0.0), 0.5)) === 0.0
    @test @inferred(ccdf(LogitNormal(0.0, 0.0), 1.0)) === 0.0
    @test @inferred(ccdf(LogitNormal(0.0, 0.0), 2.0)) === 0.0

    @test @inferred(ccdf(LogitNormal(0.0, 0.0), 0.5f0)) === 0.0
    @test @inferred(ccdf(LogitNormal(0.0f0, 0.0f0), 0.5)) === 0.0
    @test @inferred(ccdf(LogitNormal(0.0f0, 0.0f0), 0.5f0)) === 0.0f0

    @test isnan(@inferred(ccdf(LogitNormal(0.0, 0.0), NaN))::Float64)
    @test isnan(@inferred(ccdf(LogitNormal(NaN, 0.0), -1.0f0))::Float64)
    @test isnan(@inferred(ccdf(LogitNormal(NaN, 0.0), 0.0f0))::Float64)
    @test isnan(@inferred(ccdf(LogitNormal(NaN, 0.0), 0.5f0))::Float64)
    @test isnan(@inferred(ccdf(LogitNormal(NaN, 0.0), 1.0f0))::Float64)
    @test isnan(@inferred(ccdf(LogitNormal(NaN, 0.0), 2.0f0))::Float64)

    @test isnan(@inferred(ccdf(LogitNormal(NaN32, 0.0f0), -1.0f0))::Float32)
    @test isnan(@inferred(ccdf(LogitNormal(NaN32, 0.0f0), 0.0f0))::Float32)
    @test isnan(@inferred(ccdf(LogitNormal(NaN32, 0.0f0), 0.5f0))::Float32)
    @test isnan(@inferred(ccdf(LogitNormal(NaN32, 0.0f0), 1.0f0))::Float32)
    @test isnan(@inferred(ccdf(LogitNormal(NaN32, 0.0f0), 2.0f0))::Float32)

    @test @inferred(ccdf(LogitNormal(0 // 1, 0 // 1), -1 // 1)) === 1.0
    @test @inferred(ccdf(LogitNormal(0 // 1, 0 // 1), 0 // 1)) === 1.0
    @test @inferred(ccdf(LogitNormal(0 // 1, 0 // 1), 1 // 2)) === 0.0
    @test @inferred(ccdf(LogitNormal(0 // 1, 0 // 1), 1 // 1)) === 0.0
    @test @inferred(ccdf(LogitNormal(0 // 1, 0 // 1), 2 // 1)) === 0.0
    @test isnan(@inferred(ccdf(LogitNormal(0 // 1, 0 // 1), NaN))::Float64)

    @test @inferred(ccdf(LogitNormal(0.0, 0.0), BigInt(-1))) == big(1.0)
    @test @inferred(ccdf(LogitNormal(0.0, 0.0), BigInt(0))) == big(1.0)
    @test @inferred(ccdf(LogitNormal(0.0, 0.0), BigInt(1)//BigInt(2))) == big(0.0)
    @test @inferred(ccdf(LogitNormal(0.0, 0.0), BigInt(1))) == big(0.0)
    @test @inferred(ccdf(LogitNormal(0.0, 0.0), BigInt(2))) == big(0.0)
    @test @inferred(ccdf(LogitNormal(0.0, 0.0), BigFloat(-1))) == big(1.0)
    @test @inferred(ccdf(LogitNormal(0.0, 0.0), BigFloat(0))) == big(1.0)
    @test @inferred(ccdf(LogitNormal(0.0, 0.0), BigFloat(1//2))) == big(0.0)
    @test @inferred(ccdf(LogitNormal(0.0, 0.0), BigFloat(1))) == big(0.0)
    @test @inferred(ccdf(LogitNormal(0.0, 0.0), BigFloat(2))) == big(0.0)
    @test isnan(@inferred(ccdf(LogitNormal(0.0, 0.0), big(NaN)))::BigFloat)

    # logccdf
    @test @inferred(logccdf(LogitNormal(0.0, 0.0), -1.0)) === -0.0
    @test @inferred(logccdf(LogitNormal(0.0, 0.0), 0.0)) === -0.0
    @test @inferred(logccdf(LogitNormal(0.0, 0.0), 0.5)) === -Inf
    @test @inferred(logccdf(LogitNormal(0.0, 0.0), 1.0)) === -Inf
    @test @inferred(logccdf(LogitNormal(0.0, 0.0), 2.0)) === -Inf

    @test @inferred(logccdf(LogitNormal(0.0, 0.0), 0.5f0)) === -Inf
    @test @inferred(logccdf(LogitNormal(0.0f0, 0.0f0), 0.5)) === -Inf
    @test @inferred(logccdf(LogitNormal(0.0f0, 0.0f0), 0.5f0)) === -Inf32

    @test isnan(@inferred(logccdf(LogitNormal(0.0, 0.0), NaN))::Float64)
    @test isnan(@inferred(logccdf(LogitNormal(NaN, 0.0), -1.0f0))::Float64)
    @test isnan(@inferred(logccdf(LogitNormal(NaN, 0.0), 0.0f0))::Float64)
    @test isnan(@inferred(logccdf(LogitNormal(NaN, 0.0), 0.5f0))::Float64)
    @test isnan(@inferred(logccdf(LogitNormal(NaN, 0.0), 1.0f0))::Float64)
    @test isnan(@inferred(logccdf(LogitNormal(NaN, 0.0), 2.0f0))::Float64)

    @test isnan(@inferred(logccdf(LogitNormal(NaN32, 0.0f0), -1.0f0))::Float32)
    @test isnan(@inferred(logccdf(LogitNormal(NaN32, 0.0f0), 0.0f0))::Float32)
    @test isnan(@inferred(logccdf(LogitNormal(NaN32, 0.0f0), 0.5f0))::Float32)
    @test isnan(@inferred(logccdf(LogitNormal(NaN32, 0.0f0), 1.0f0))::Float32)
    @test isnan(@inferred(logccdf(LogitNormal(NaN32, 0.0f0), 2.0f0))::Float32)

    @test @inferred(logccdf(LogitNormal(0 // 1, 0 // 1), -1 // 1)) === -0.0
    @test @inferred(logccdf(LogitNormal(0 // 1, 0 // 1), 0 // 1)) === -0.0
    @test @inferred(logccdf(LogitNormal(0 // 1, 0 // 1), 1 // 2)) === -Inf
    @test @inferred(logccdf(LogitNormal(0 // 1, 0 // 1), 1 // 1)) === -Inf
    @test @inferred(logccdf(LogitNormal(0 // 1, 0 // 1), 2 // 1)) === -Inf
    @test isnan(@inferred(logccdf(LogitNormal(0 // 1, 0 // 1), NaN))::Float64)

    @test @inferred(logccdf(LogitNormal(0.0, 0.0), BigInt(-1))) == big(0.0)
    @test @inferred(logccdf(LogitNormal(0.0, 0.0), BigInt(0))) == big(0.0)
    @test @inferred(logccdf(LogitNormal(0.0, 0.0), BigInt(1)//BigInt(2))) == big(-Inf)
    @test @inferred(logccdf(LogitNormal(0.0, 0.0), BigInt(1))) == big(-Inf)
    @test @inferred(logccdf(LogitNormal(0.0, 0.0), BigInt(2))) == big(-Inf)
    @test @inferred(logccdf(LogitNormal(0.0, 0.0), BigFloat(-1))) == big(0.0)
    @test @inferred(logccdf(LogitNormal(0.0, 0.0), BigFloat(0))) == big(0.0)
    @test @inferred(logccdf(LogitNormal(0.0, 0.0), BigFloat(1//2))) == big(-Inf)
    @test @inferred(logccdf(LogitNormal(0.0, 0.0), BigFloat(1))) == big(-Inf)
    @test @inferred(logccdf(LogitNormal(0.0, 0.0), BigFloat(2))) == big(-Inf)
    @test isnan(@inferred(logccdf(LogitNormal(0.0, 0.0), big(NaN)))::BigFloat)

    # quantile
    @test @inferred(quantile(LogitNormal(1.0, 0.0), 0.0f0)) === 0.0
    @test @inferred(quantile(LogitNormal(1.0, 0.0f0), 1.0)) === 1.0
    @test @inferred(quantile(LogitNormal(1.0f0, 0.0), 0.5)) === logistic(1.0)
    @test isnan(@inferred(quantile(LogitNormal(1.0f0, 0.0), NaN))::Float64)
    @test @inferred(quantile(LogitNormal(1.0f0, 0.0f0), 0.0f0)) === 0.0f0
    @test @inferred(quantile(LogitNormal(1.0f0, 0.0f0), 1.0f0)) === 1.0f0
    @test @inferred(quantile(LogitNormal(1.0f0, 0.0f0), 0.5f0)) === logistic(1.0f0)
    @test isnan(@inferred(quantile(LogitNormal(1.0f0, 0.0f0), NaN32))::Float32)
    @test @inferred(quantile(LogitNormal(1//1, 0//1), 1//2)) === logistic(1.0)

    # invlogcdf
    @test @inferred(invlogcdf(LogitNormal(1.0, 0.0), -Inf32)) === 0.0
    @test @inferred(invlogcdf(LogitNormal(1.0, 0.0f0), 0.0)) === 1.0
    @test @inferred(invlogcdf(LogitNormal(1.0f0, 0.0), -log(2.0))) === logistic(1.0)
    @test isnan(@inferred(invlogcdf(LogitNormal(1.0f0, 0.0), NaN))::Float64)
    @test @inferred(invlogcdf(LogitNormal(1.0f0, 0.0f0), -Inf32)) === 0.0f0
    @test @inferred(invlogcdf(LogitNormal(1.0f0, 0.0f0), 0.0f0)) === 1.0f0
    @test @inferred(invlogcdf(LogitNormal(1.0f0, 0.0f0), -log(2f0))) === logistic(1.0f0)
    @test isnan(@inferred(invlogcdf(LogitNormal(1.0f0, 0.0f0), NaN32))::Float32)
    @test @inferred(invlogcdf(LogitNormal(1//1, 0//1), -log(2.0))) === logistic(1.0)

    # cquantile
    @test @inferred(cquantile(LogitNormal(1.0, 0.0), 0.0f0)) === 1.0
    @test @inferred(cquantile(LogitNormal(1.0, 0.0f0), 1.0)) === 0.0
    @test @inferred(cquantile(LogitNormal(1.0f0, 0.0), 0.5)) === logistic(1.0)
    @test isnan(@inferred(cquantile(LogitNormal(1.0f0, 0.0), NaN))::Float64)
    @test @inferred(cquantile(LogitNormal(1.0f0, 0.0f0), 0.0f0)) === 1.0f0
    @test @inferred(cquantile(LogitNormal(1.0f0, 0.0f0), 1.0f0)) === 0.0f0
    @test @inferred(cquantile(LogitNormal(1.0f0, 0.0f0), 0.5f0)) === logistic(1.0f0)
    @test isnan(@inferred(cquantile(LogitNormal(1.0f0, 0.0f0), NaN32))::Float32)
    @test @inferred(cquantile(LogitNormal(1//1, 0//1), 1//2)) === logistic(1.0)

    # invlogccdf
    @test @inferred(invlogccdf(LogitNormal(1.0, 0.0), -Inf32)) === 1.0
    @test @inferred(invlogccdf(LogitNormal(1.0, 0.0f0), 0.0)) === 0.0
    @test @inferred(invlogccdf(LogitNormal(1.0f0, 0.0), -log(2.0))) === logistic(1.0)
    @test isnan(@inferred(invlogccdf(LogitNormal(1.0f0, 0.0), NaN))::Float64)
    @test @inferred(invlogccdf(LogitNormal(1.0f0, 0.0f0), -Inf32)) === 1.0f0
    @test @inferred(invlogccdf(LogitNormal(1.0f0, 0.0f0), 0.0f0)) === 0.0f0
    @test @inferred(invlogccdf(LogitNormal(1.0f0, 0.0f0), -log(2.0f0))) === logistic(1.0f0)
    @test isnan(@inferred(invlogccdf(LogitNormal(1.0f0, 0.0f0), NaN32))::Float32)
    @test @inferred(invlogccdf(LogitNormal(1//1, 0//1), -log(2.0))) === logistic(1.0)
end

@testset "LogitNormal: gradlogpdf" begin
    @test @inferred(gradlogpdf(LogitNormal(0.0, 1.0), -1.0)) === 0.0
    @test @inferred(gradlogpdf(LogitNormal(0.0, 1.0), 0.0)) === 0.0
    @test @inferred(gradlogpdf(LogitNormal(0.0, 1.0), logistic(-1)))::Float64 ≈ 2 * (exp(-1) + 1)
    @test @inferred(gradlogpdf(LogitNormal(0.0, 1.0), 0.5)) === 0.0
    @test @inferred(gradlogpdf(LogitNormal(0.0, 1.0), 1.0)) === 0.0
    @test @inferred(gradlogpdf(LogitNormal(0.0, 1.0), 2.0)) === 0.0

    @test @inferred(gradlogpdf(LogitNormal(0.0, 1.0), 0.5f0)) === 0.0
    @test @inferred(gradlogpdf(LogitNormal(0.0f0, 1.0f0), 0.5)) === 0.0
    @test @inferred(gradlogpdf(LogitNormal(0.0f0, 1.0f0), 0.5f0)) === 0.0f0

    @test @inferred(gradlogpdf(LogitNormal(0 // 1, 1 // 1), -1 // 1)) === 0.0
    @test @inferred(gradlogpdf(LogitNormal(0 // 1, 1 // 1), 0 // 1)) === 0.0
    @test @inferred(gradlogpdf(LogitNormal(0 // 1, 1 // 1), 1 // 2)) === 0.0
    @test @inferred(gradlogpdf(LogitNormal(0 // 1, 1 // 1), 1 // 1)) === 0.0
    @test @inferred(gradlogpdf(LogitNormal(0 // 1, 1 // 1), 2 // 1)) === 0.0
    @test isnan(@inferred(gradlogpdf(LogitNormal(0 // 1, 1 // 1), NaN))::Float64)

    @test @inferred(gradlogpdf(LogitNormal(0.0, 1.0), BigInt(-1))) == big(0.0)
    @test @inferred(gradlogpdf(LogitNormal(0.0, 1.0), BigInt(0))) == big(0.0)
    @test @inferred(gradlogpdf(LogitNormal(0.0, 1.0), BigInt(1)//BigInt(2))) == big(0.0)
    @test @inferred(gradlogpdf(LogitNormal(0.0, 1.0), BigInt(1))) == big(0.0)
    @test @inferred(gradlogpdf(LogitNormal(0.0, 1.0), BigInt(2))) == big(0.0)
    @test @inferred(gradlogpdf(LogitNormal(0.0, 1.0), BigFloat(-1))) == big(0.0)
    @test @inferred(gradlogpdf(LogitNormal(0.0, 1.0), BigFloat(0))) == big(0.0)
    @test @inferred(gradlogpdf(LogitNormal(0.0, 1.0), logistic(BigFloat(-1)))) ≈ 2 * (exp(big(-1)) + 1)
    @test @inferred(gradlogpdf(LogitNormal(0.0, 1.0), BigFloat(1//2))) == big(0.0)
    @test @inferred(gradlogpdf(LogitNormal(0.0, 1.0), BigFloat(1))) == big(0.0)
    @test @inferred(gradlogpdf(LogitNormal(0.0, 1.0), BigFloat(2))) == big(0.0)
    @test isnan(@inferred(gradlogpdf(LogitNormal(0.0, 1.0), big(NaN)))::BigFloat)
end

@testset "Logitnormal Sampling Tests" begin
    for d in [
        LogitNormal(-2, 3),
        LogitNormal(0, 0.2)
    ]
        test_distr(d, 10^6)
    end
end
