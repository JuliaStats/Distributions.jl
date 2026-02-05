# Tests for Multivariate Hypergeometric

using Distributions
using Test

@testset "Multivariate Hypergeometric" begin
    @test_throws DomainError MvHypergeometric([5, 3, -2], 4)
    @test_throws ArgumentError MvHypergeometric([5, 3], 10)

    m = [5, 3, 2]
    n = 4
    d = MvHypergeometric(m, n)
    @test length(d) == 3
    @test d.n == n
    @test d.m == m
    @test ncategories(d) == length(m)
    @test params(d) == (m, n)
    @test partype(d) == Int

    @test mean(d) ≈ [2.0, 1.2, 0.8]
    @test var(d) ≈ [2 / 3, 56 / 100, 32 / 75]

    covmat = cov(d)
    @test covmat ≈ (8 / 3) .* [1/4 -3/20 -1/10; -3/20 21/100 -3/50; -1/10 -3/50 4/25]

    @test insupport(d, [2, 1, 1])
    @test !insupport(d, [3, 2, 1])
    @test !insupport(d, [0, 0, 4])


    # random sampling
    x = rand(d)
    @test isa(x, Vector{Int})
    @test sum(x) == n
    @test all(x .>= 0)
    @test all(x .<= m)
    @test insupport(d, x)

    x = rand(d, 100)
    @test isa(x, Matrix{Int})
    @test all(sum(x, dims=1) .== n)
    @test all(x .>= 0)
    @test all(x .<= m)
    @test all(insupport(d, x))

    # random sampling with many catergories
    m = [20, 2, 2, 2, 1, 1, 1]
    n = 5
    d2 = MvHypergeometric(m, n)
    x = rand(d2)
    @test isa(x, Vector{Int})
    @test sum(x) == n
    @test all(x .>= 0)
    @test all(x .<= m)
    @test insupport(d2, x)

    # random sampling with a large category
    m = [2, 1000]
    n = 5
    d3 = MvHypergeometric(m, n)
    x = rand(d3)
    @test isa(x, Vector{Int})
    @test sum(x) == n
    @test all(x .>= 0)
    @test all(x .<= m)
    @test insupport(d3, x)

    # log pdf
    x = [2, 1, 1]
    @test pdf(d, x) ≈ 2 / 7
    @test logpdf(d, x) ≈ log(2 / 7)
    @test logpdf(d, x) ≈ log(pdf(d, x))
    @test logpdf(d, [2.5, 0.5, 1]) == -Inf

    x = rand(d, 100)
    pv = pdf(d, x)
    lp = logpdf(d, x)
    for i in 1:size(x, 2)
        @test pv[i] ≈ pdf(d, x[:, i])
        @test lp[i] ≈ logpdf(d, x[:, i])
    end

    # test degenerate cases
    d1 = MvHypergeometric([1], 1)
    @test logpdf(d1, [1]) ≈ 0
    @test logpdf(d1, [0]) == -Inf
    d2 = MvHypergeometric([2, 0], 1)
    @test logpdf(d2, [1, 0]) ≈ 0
    @test logpdf(d2, [0, 1]) == -Inf

    d3 = MvHypergeometric([5, 0, 0, 0], 3)
    @test logpdf(d3, [3, 0, 0, 0]) ≈ 0
    @test logpdf(d3, [2, 1, 0, 0]) == -Inf
    @test logpdf(d3, [2, 0, 0, 0]) == -Inf

    # behavior with n = 0
    d0 = MvHypergeometric([5, 3, 2], 0)
    @test logpdf(d0, [0, 0, 0]) ≈ 0
    @test logpdf(d0, [1, 0, 0]) == -Inf

    @test rand(d0) == [0, 0, 0]
    @test mean(d0) == [0.0, 0.0, 0.0]
    @test var(d0) == [0.0, 0.0, 0.0]
    @test insupport(d0, [0, 0, 0])
    @test !insupport(d0, [1, 0, 0])
    @test length(d0) == 3

    # compare with hypergeometric
    ns = 3
    nf = 5
    n = 4
    dh1 = MvHypergeometric([ns, nf], n)
    dh2 = Hypergeometric(ns, nf, n)

    x = 2
    @test pdf(dh1, [x, n - x]) ≈ pdf(dh2, x)
    x = 3
    @test pdf(dh1, [x, n - x]) ≈ pdf(dh2, x)

    # comparing marginals to hypergeometric
    m = [5, 3, 2]
    n = 4
    d = MvHypergeometric(m, n)
    dh = Hypergeometric(m[1], sum(m[2:end]), n)
    x1 = 2
    @test pdf(dh, x1) ≈ sum([pdf(d, [x1, x2, n - x1 - x2]) for x2 in 0:m[2]])

    # comparing conditionals to hypergeometric
    x1 = 2
    dh = Hypergeometric(m[2], m[3], n - x1)
    q = sum([pdf(d, [x1, x2, n - x1 - x2]) for x2 in 0:m[2]])
    for x2 = 0:m[2]
        @test pdf(dh, x2) ≈ pdf(d, [x1, x2, n - x1 - x2]) / q
    end
end