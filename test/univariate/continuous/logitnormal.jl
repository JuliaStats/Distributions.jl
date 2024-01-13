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
