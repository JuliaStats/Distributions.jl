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

@testset "LKJ integrating constant" begin
    #  =============
    #  odd non-uniform
    #  =============
    d = 5
    η = 2.3
    lkj = LKJ(d, η)
    @test Distributions.lkj_vine_logc0(d, η) ≈ Distributions.lkj_onion_logc0(d, η)
    @test Distributions.lkj_onion_logc0(d, η) ≈ Distributions.lkj_logc0_alt(d, η)
    @test lkj.logc0 == Distributions.lkj_onion_logc0(d, η)
    #  =============
    #  odd uniform
    #  =============
    d = 5
    η = 1.0
    lkj = LKJ(d, η)
    @test Distributions.lkj_vine_logc0(d, η) ≈ Distributions.lkj_onion_logc0(d, η)
    @test Distributions.lkj_onion_logc0(d, η) ≈ Distributions.lkj_onion_logc0_uniform_odd(d)
    @test Distributions.lkj_vine_logc0(d, η) ≈ Distributions.lkj_vine_logc0_uniform(d)
    @test Distributions.lkj_onion_logc0(d, η) ≈ Distributions.lkj_logc0_alt(d, η)
    @test lkj.logc0 == Distributions.lkj_onion_logc0_uniform_odd(d)
    #  =============
    #  even non-uniform
    #  =============
    d = 6
    η = 2.3
    lkj = LKJ(d, η)
    @test Distributions.lkj_vine_logc0(d, η) ≈ Distributions.lkj_onion_logc0(d, η)
    @test Distributions.lkj_onion_logc0(d, η) ≈ Distributions.lkj_logc0_alt(d, η)
    @test lkj.logc0 == Distributions.lkj_onion_logc0(d, η)
    #  =============
    #  even uniform
    #  =============
    d = 6
    η = 1.0
    lkj = LKJ(d, η)
    @test Distributions.lkj_vine_logc0(d, η) ≈ Distributions.lkj_onion_logc0(d, η)
    @test Distributions.lkj_onion_logc0(d, η) ≈ Distributions.lkj_onion_logc0_uniform_even(d)
    @test Distributions.lkj_vine_logc0(d, η) ≈ Distributions.lkj_vine_logc0_uniform(d)
    @test Distributions.lkj_onion_logc0(d, η) ≈ Distributions.lkj_logc0_alt(d, η)
    @test lkj.logc0 == Distributions.lkj_onion_logc0_uniform_even(d)
end
