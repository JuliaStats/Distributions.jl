# Tests for Multivariate Hypergeometric

using Distributions, Random
using Test

@testset "Multivariate Hypergeometric" begin 
    m = [5, 3, 2]
    n = 4
    d = MvHypergeometric(m, n)
    
    @test length(d) == 3
    @test d.n == n
    @test nelements(d) == m
    @test ncategories(d) == length(m)
    
    @test mean(d) ≈ [2.0, 1.2, 0.8]
    
    @test var(d) ≈ [2/3, 56/100, 32/75]
    
    covmat = cov(d)
    @test covmat ≈ (8/3) .* [1/4 -3/20 -1/10; -3/20 21/100 -3/50; -1/10 -3/50 4/25]
    
    @test insupport(d, [2, 1, 1])
    @test !insupport(d, [3, 2, 1])
    @test !insupport(d, [0, 0, 4])
    
    # random sampling
    rng = MersenneTwister(123)
    x = rand(rng, d)
    @test isa(x, Vector{Int})
    @test sum(x) == n
    @test insupport(d, x)
    
    x = rand(rng, d, 100)
    @test isa(x, Matrix{Int})
    @test all(sum(x, dims=1) .== n)
    @test all(x .>= 0)
    @test all(x .<= m)
    @test all(insupport(d, x))

    # log pdf
    x = [2, 1, 1]
    @test pdf(d, x) ≈ 2/7
    @test logpdf(d, x) ≈ log(2/7)
    @test logpdf(d, x) ≈ log(pdf(d, x))

    x = rand(rng, d, 100)
    pv = pdf(d, x)
    lp = logpdf(d, x)
    for i in 1 : size(x, 2)
        @test pv[i] ≈ pdf(d, x[:,i])
        @test lp[i] ≈ logpdf(d, x[:,i])
    end

    # test degenerate cases of logpdf
    d1 = MvHypergeometric([1], 1)
    @test logpdf(d1, [1]) ≈ 0
    @test logpdf(d1, [0]) == -Inf
    d2 = MvHypergeometric([2, 0], 1)
    @test logpdf(d2, [1, 0]) ≈ 0
    @test logpdf(d2, [0, 1]) == -Inf

    # behavior with n = 0
    d0 = MvHypergeometric([5, 3, 2], 0)
    @test logpdf(d0, [0, 0, 0]) ≈ 0
    @test logpdf(d0, [1, 0, 0]) == -Inf
 
    @test rand(rng, d0) == [0, 0, 0]
    @test mean(d0) == [0.0, 0.0, 0.0]
    @test var(d0) == [0.0, 0.0, 0.0]
    @test insupport(d0, [0, 0, 0])
    @test !insupport(d0, [1, 0, 0])
    @test length(d0) == 3
end