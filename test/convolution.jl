using Distributions
using FillArrays

using LinearAlgebra
using Test

@testset "discrete univariate" begin
    @testset "Bernoulli" begin
        d1 = Bernoulli(0.1)
        d2 = @inferred(convolve(d1, d1))
        @test d2 isa Binomial
        @test d2.n == 2
        @test d2.p == 0.1

        # cannot convolve a Binomial with a Bernoulli
        @test_throws MethodError convolve(d1, d2)

        # only works if p1 ≈ p2
        d3 = Bernoulli(0.2)
        @test_throws ArgumentError convolve(d1, d3)
    end

    @testset "Binomial" begin
        d1 = Binomial(2, 0.1)
        d2 = Binomial(5, 0.1)
        d3 = @inferred(convolve(d1, d2))
        @test d3 isa Binomial
        @test d3.n == 7
        @test d3.p == 0.1

        # only works if p1 ≈ p2
        d4 = Binomial(2, 0.2)
        @test_throws ArgumentError convolve(d1, d4)
    end

    @testset "NegativeBinomial" begin
        d1 = NegativeBinomial(4, 0.1)
        d2 = NegativeBinomial(1, 0.1)
        d3 = @inferred(convolve(d1, d2))
        @test d3 isa NegativeBinomial
        @test d3.r == 5
        @test d3.p == 0.1

        d4 = NegativeBinomial(1, 0.2)
        @test_throws ArgumentError convolve(d1, d4)
    end

    @testset "Geometric" begin
        d1 = Geometric(0.2)
        d2 = convolve(d1, d1)
        @test d2 isa NegativeBinomial
        @test d2.p == 0.2

        # cannot convolve a Geometric with a NegativeBinomial
        @test_throws MethodError convolve(d1, d2)

        # only works if p1 ≈ p2
        d3 = Geometric(0.5)
        @test_throws ArgumentError convolve(d1, d3)
    end

    @testset "Poisson" begin
        d1 = Poisson(0.1)
        d2 = Poisson(0.4)
        d3 = @inferred(convolve(d1, d2))
        @test d3 isa Poisson
        @test d3.λ == 0.5
    end
end

@testset "continuous univariate" begin
    @testset "Gaussian" begin
        d1 = Normal(0.1, 0.2)
        d2 = Normal(0.25, 1.7)
        d3 = @inferred(convolve(d1, d2))
        @test d3 isa Normal
        @test d3.μ == 0.35
        @test d3.σ == hypot(0.2, 1.7)
    end

    @testset "Cauchy" begin
        d1 = Cauchy(0.2, 0.7)
        d2 = Cauchy(1.9, 0.8)
        d3 = @inferred(convolve(d1, d2))
        @test d3 isa Cauchy
        @test d3.μ == 2.1
        @test d3.σ == 1.5
    end

    @testset "Chisq" begin
        d1 = Chisq(0.1)
        d2 = Chisq(0.3)
        d3 = @inferred(convolve(d1, d2))
        @test d3 isa Chisq
        @test d3.ν == 0.4
    end

    @testset "Exponential" begin
        d1 = Exponential(0.7)
        d2 = @inferred(convolve(d1, d1))
        @test d2 isa Gamma
        @test d2.α == 2
        @test d2.θ == 0.7

        # cannot convolve an Exponential with a Gamma
        @test_throws MethodError convolve(d1, d2)

        # only works if θ1 ≈ θ2
        d3 = Exponential(0.2)
        @test_throws ArgumentError convolve(d1, d3)
    end

    @testset "Gamma" begin
        d1 = Gamma(0.1, 1.7)
        d2 = Gamma(0.5, 1.7)
        d3 = @inferred(convolve(d1, d2))
        @test d3 isa Gamma
        @test d3.α == 0.6
        @test d3.θ == 1.7

        # only works if θ1 ≈ θ4
        d4 = Gamma(1.2, 0.4)
        @test_throws ArgumentError convolve(d1, d4)
    end
end

@testset "continuous multivariate" begin
    @testset "iso-/diag-normal" begin
        in1 = MvNormal([1.2, 0.3], 2 * I)
        in2 = MvNormal([-2.0, 6.9], 0.5 * I)

        zmin1 = MvNormal(Zeros(2), 1.9 * I)
        zmin2 = MvNormal(Diagonal(Fill(5.2, 2)))

        dn1 = MvNormal([0.0, 4.7], Diagonal([0.1, 1.8]))
        dn2 = MvNormal([-3.4, 1.2], Diagonal([3.2, 0.2]))

        zmdn1 = MvNormal(Diagonal([1.2, 0.3]))
        zmdn2 = MvNormal(Diagonal([0.8, 1.0]))

        m1 = Symmetric(rand(2,2))
        fn1 = MvNormal(ones(2), m1^2)

        m2 = Symmetric(rand(2,2))
        fn2 = MvNormal([2.1, 0.4], m2^2)

        m3 = Symmetric(rand(2,2))
        zm1 = MvNormal(m3^2)

        m4 = Symmetric(rand(2,2))
        zm2 = MvNormal(m4^2)

        dist_list = (in1, in2, zmin1, zmin2, dn1, dn2, zmdn1, zmdn2, fn1, fn2, zm1, zm2)

        for (d1, d2) in Iterators.product(dist_list, dist_list)
            d3  = @inferred(convolve(d1, d2))
            @test d3 isa MvNormal
            @test d3.μ == d1.μ .+ d2.μ
            @test Matrix(d3.Σ) == Matrix(d1.Σ + d2.Σ)  # isequal not defined for PDMats
        end

        # erroring
        in3 = MvNormal([1, 2, 3], 0.2 * I)
        @test_throws ArgumentError convolve(in1, in3)

        m5 = Symmetric(rand(3, 3))
        fn3 = MvNormal(zeros(3), m5^2)
        @test_throws ArgumentError convolve(fn1, fn3)
    end
end
