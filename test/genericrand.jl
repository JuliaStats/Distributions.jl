using Distributions
using LinearAlgebra
using Random
using Test

@testset "genericrand.jl" begin
    gmm = UnivariateGMM([0.0, 1.0], [1.0, 2.0], Categorical([0.3, 0.7]))
    dists = [
        # univariate
        Normal(), Normal(0f0, 1f0), Poisson(3.0), gmm,
        # multivariate
        MvNormal(zeros(2), I), Dirichlet(3, 1.0), Multinomial(4, [0.2, 0.8]),
        product_distribution(fill(Normal(), 3)),
        product_distribution(Normal(), Exponential()),
        # matrix-variate
        reshape(MvNormal(zeros(4), I), 2, 2), Wishart(3.0, Matrix(1.0I, 2, 2)), LKJ(3, 1.0),
        # other variate forms
        LKJCholesky(3, 1.0), product_distribution((x=Normal(), y=Poisson())),
    ]

    @testset "multiple samples: $(typeof(d))" for d in dists
        x = rand(d)
        for dims in ((3,), (2, 3), (0,), (0, 2))
            xs = rand(d, dims)
            @test xs isa Array{typeof(x),length(dims)}
            @test size(xs) == dims
        end
    end

    @testset "dimension mismatch: $(typeof(d))" for d in dists
        d isa Distribution{<:ArrayLikeVariate} && !(d isa UnivariateDistribution) || continue
        x = rand(d)
        # one sample
        @test_throws DimensionMismatch rand!(d, similar(x, size(x) .+ 1))
        # multiple samples
        @test_throws DimensionMismatch rand!(d, similar(x, (size(x) .+ 1)..., 2))
        @test_throws DimensionMismatch rand!(d, [similar(x, size(x) .+ 1)], false)
    end

    @testset "dimension mismatch: LKJCholesky" begin
        @test_throws DimensionMismatch rand!(LKJCholesky(3, 1.0), rand(LKJCholesky(2, 1.0)))
    end

    @testset "UnivariateGMM sampler" begin
        @test rand(sampler(gmm)) isa Float64
        @test rand(Xoshiro(1), sampler(gmm), 3) isa Vector{Float64}
    end

    @testset "multiple samples of tuple product distribution" begin
        d = product_distribution(Normal(), Normal(0f0, 1f0))
        @test @inferred(rand(d)) isa Vector{Float64}
        @test @inferred(rand(d, 3)) isa Matrix{Float64}
        @test size(rand(d, 3)) == (2, 3)
    end
end
