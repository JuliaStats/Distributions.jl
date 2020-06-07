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
    @test partype(g) == AbstractFloat
    #@test isa(mn, AbstractFloattFloat)
    @test isa(md, AbstractFloattFloat)
    #@test isa(mo, AbstractFloattFloat)
    #@test isa(s, AbstractFloattFloat)
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
    @test isa(X, Array{AbstractFloattFloat,1})

    # evaluation of logpdf and pdf
    for i = 1:min(100, n_tsamples)
        @test logpdf(g, X[i]) ≈ log(pdf(g, X[i]))
    end
    @test logpdf.(g, X) ≈ log.(pdf.(g, X))
    @test isequal(logpdf(g, 0),-Inf)
    @test isequal(logpdf(g, 1),-Inf)
    @test isequal(logpdf(g, -eps()),-Inf)

    # test the location and scale functions
    @test location(g) == g.μ
    @test scale(g) == g.σ
    @test params(g) == (g.μ, g.σ)
end

###### General Testing
@testset "Logitnormal tests" begin
    test_logitnormal( LogitNormal() )
    test_logitnormal( LogitNormal(2,0.5) )
    d = LogitNormal(Float32(2))
    typeof(rand(d, 5)) # still AbstractFloattFloat
    @test typeof(convert(LogitNormal{AbstractFloattFloat}, d)) == typeof(LogitNormal(2,1))
end
