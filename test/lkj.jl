using Distributions, Random
using Test, LinearAlgebra, PDMats, Statistics, HypothesisTests


d = 4
η = abs(3randn())
η₀ = 1

G = LKJ(d, η)
F = LKJ(d, η₀)

@testset "LKJ construction errors" begin
    @test_throws ArgumentError LKJ(-1, η)
    @test_throws ArgumentError LKJ(d, -1)
    @test LKJ(-1, η, check_args = false) isa LKJ{typeof(η), typeof(-1)}
end

@testset "LKJ params" begin
    η̃ = params(G)
    η̃₀ = params(F)
    @test η̃ == η
    @test η̃₀ == η₀
end

@testset "LKJ dim" begin
    @test dim(G) == d
    @test dim(F) == d
end

@testset "LKJ size" begin
    @test size(G) == (d, d)
    @test size(F) == (d, d)
end

@testset "LKJ rank" begin
    @test rank(G) == d
    @test rank(F) == d
    @test rank(G) == rank(rand(G))
    @test rank(F) == rank(rand(F))
end

@testset "LKJ insupport" begin
    @test insupport(G, rand(G))
    @test insupport(F, rand(F))

    @test !insupport(G, rand(G) + rand(G) * im)
    @test !insupport(G, randn(d, d + 1))
    @test !insupport(G, randn(d, d))
end

@testset "LKJ mode" begin
    @test mode(LKJ(5, 1.5)) == mean(LKJ(5, 1.5))
    @test_throws ArgumentError mode( LKJ(5, 0.5) )
end

@testset "LKJ sample moments" begin
    @test isapprox(mean(rand(G, 100000)), mean(G) , atol = 0.1)
    @test isapprox(var(rand(G, 100000)), var(G) , atol = 0.1)
end

@testset "LKJ marginals" begin
    M = 10000
    α = 0.05
    L = sum(1:(d - 1))
    ρ = Distributions._marginal(G)
    mymats = zeros(d, d, M)
    for m in 1:M
        mymats[:, :, m] = rand(G)
    end
    for i in 1:d
        for j in 1:i-1
            kstest = ExactOneSampleKSTest(mymats[i, j, :], ρ)
            @test pvalue(kstest) >= α / L  #  multiple comparisons
        end
    end
end

@testset "LKJ conversion" for elty in (Float32, Float64, BigFloat)
    Gel1 = convert(LKJ{elty}, G)
    Gel2 = convert(LKJ{elty}, G.d, G.η, G.logc0)

    @test Gel1 isa LKJ{elty, typeof(d)}
    @test Gel2 isa LKJ{elty, typeof(d)}
    @test partype(Gel1) == elty
    @test partype(Gel2) == elty
end

@testset "check d = 1 edge case" begin
    lkj = LKJ(1, 2abs(randn()))
    @test var(lkj) == zeros(1, 1)
    @test rand(lkj) == ones(1, 1)
end

@testset "LKJ integrating constant" begin
    #  =============
    #  odd non-uniform
    #  =============
    d = 5
    η = 2.3
    lkj = LKJ(d, η)
    @test Distributions.lkj_vine_loginvconst(d, η) ≈ Distributions.lkj_onion_loginvconst(d, η)
    @test Distributions.lkj_onion_loginvconst(d, η) ≈ Distributions.lkj_loginvconst_alt(d, η)
    @test lkj.logc0 == -Distributions.lkj_onion_loginvconst(d, η)
    #  =============
    #  odd uniform
    #  =============
    d = 5
    η = 1.0
    lkj = LKJ(d, η)
    @test Distributions.lkj_vine_loginvconst(d, η) ≈ Distributions.lkj_onion_loginvconst(d, η)
    @test Distributions.lkj_onion_loginvconst(d, η) ≈ Distributions.lkj_onion_loginvconst_uniform_odd(d)
    @test Distributions.lkj_vine_loginvconst(d, η) ≈ Distributions.lkj_vine_loginvconst_uniform(d)
    @test Distributions.lkj_onion_loginvconst(d, η) ≈ Distributions.lkj_loginvconst_alt(d, η)
    @test Distributions.lkj_onion_loginvconst(d, η) ≈ Distributions.corr_logvolume(d)
    @test lkj.logc0 == -Distributions.lkj_onion_loginvconst_uniform_odd(d)
    #  =============
    #  even non-uniform
    #  =============
    d = 6
    η = 2.3
    lkj = LKJ(d, η)
    @test Distributions.lkj_vine_loginvconst(d, η) ≈ Distributions.lkj_onion_loginvconst(d, η)
    @test Distributions.lkj_onion_loginvconst(d, η) ≈ Distributions.lkj_loginvconst_alt(d, η)
    @test lkj.logc0 == -Distributions.lkj_onion_loginvconst(d, η)
    #  =============
    #  even uniform
    #  =============
    d = 6
    η = 1.0
    lkj = LKJ(d, η)
    @test Distributions.lkj_vine_loginvconst(d, η) ≈ Distributions.lkj_onion_loginvconst(d, η)
    @test Distributions.lkj_onion_loginvconst(d, η) ≈ Distributions.lkj_onion_loginvconst_uniform_even(d)
    @test Distributions.lkj_vine_loginvconst(d, η) ≈ Distributions.lkj_vine_loginvconst_uniform(d)
    @test Distributions.lkj_onion_loginvconst(d, η) ≈ Distributions.lkj_loginvconst_alt(d, η)
    @test Distributions.lkj_onion_loginvconst(d, η) ≈ Distributions.corr_logvolume(d)
    @test lkj.logc0 == -Distributions.lkj_onion_loginvconst_uniform_even(d)
end

@testset "check integrating constant as a volume" begin
    #  d = 2: Lebesgue measure of the set of correlation matrices is 2.
    volume2D = 2
    @test volume2D ≈ exp( Distributions.lkj_onion_loginvconst(2, 1) )
    @test 1 / volume2D ≈ exp( LKJ(2, 1).logc0 )
    #  d = 3: Lebesgue measure of the set of correlation matrices is π²/2.
    #  See here: https://www.jstor.org/stable/2684832
    volume3D = 0.5π^2
    @test volume3D ≈ exp( Distributions.lkj_onion_loginvconst(3, 1) )
    @test 1 / volume3D ≈ exp( LKJ(3, 1).logc0 )
    #  d = 4: Lebesgue measure of the set of correlation matrices is (32/27)π².
    #  See here: https://doi.org/10.4169/amer.math.monthly.123.9.909
    volume4D = (32 / 27)*π^2
    @test volume4D ≈ exp( Distributions.lkj_onion_loginvconst(4, 1) )
    @test 1 / volume4D ≈ exp( LKJ(4, 1).logc0 )
end

@testset "check logpdf against archived Stan output" begin
    #  Compare to archived output from Stan's lkj_corr_lpdf function.
    #  https://mc-stan.org/docs/2_22/functions-reference/lkj-correlation.html
    #  https://mc-stan.org/math/db/d4f/lkj__corr__lpdf_8hpp_source.html
    R = [1 0.962395133838894 -0.436307195544856;
         0.962395133838894 1 -0.301102833786894;
        -0.436307195544856 -0.301102833786894 1]
    n = size(R, 1)
    @test isapprox(logpdf(LKJ(n, 0.5), R), -0.9874823, atol = 1e-6)
    @test isapprox(logpdf(LKJ(n, 1),   R), -1.596313, atol = 1e-6)
    @test isapprox(logpdf(LKJ(n, 3.4), R), -7.253798, atol = 1e-6)
end

@testset "importance sampling check" begin
    d = 3
    M = 20000
    f = LKJ(d, 2)
    g = LKJ(d, 1)
    h = mean(logdet.(rand(f, M)))
    ĥ = mean(logdet(R) * pdf(f, R) / pdf(g, R) for R in (rand(g) for i in 1:M))
    @test isapprox(h, ĥ, atol = 0.1)
end
