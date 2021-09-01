using Distributions
using FillArrays

using Test
using Random
using LinearAlgebra


@testset "Testing normal product distributions" begin
    Random.seed!(123456)
    N = 11

    # Construct independent distributions and `ProductDistribution` from these.
    μ = randn(N)

    ds1 = Normal.(μ, 1.0)
    d_product1 = @inferred(product_distribution(ds1))
    @test d_product1 isa Distributions.DiagNormal

    ds2 = Fill(Normal(first(μ), 1.0), N)
    d_product2 = @inferred(product_distribution(ds2))
    @test d_product2 isa MvNormal{Float64,Distributions.ScalMat{Float64},<:Fill{Float64,1}}

    # Check that methods for `ProductDistribution` are consistent.
    for (ds, d_product) in ((ds1, d_product1), (ds2, d_product2))
        @test length(d_product) == length(ds)
        @test eltype(d_product) === eltype(ds[1])
        @test mean(d_product) == mean.(ds)
        @test var(d_product) == var.(ds)
        @test cov(d_product) == Diagonal(var.(ds))
        @test entropy(d_product) ≈ sum(entropy.(ds))

        x = rand(d_product)
        @test x isa typeof(rand.(collect(ds)))
        @test length(x) == N
        @test logpdf(d_product, x) ≈ sum(logpdf.(ds, x))
    end
end

@testset "Testing generic VectorOfUnivariateDistribution" begin
    Random.seed!(123456)
    N = 11

    # Construct independent distributions and `ProductDistribution` from these.
    ubound = rand(N)

    ds1 = Uniform.(0.0, ubound)
    d_product1 = @inferred(product_distribution(ds1))
    @test d_product1 isa Distributions.VectorOfUnivariateDistribution{Continuous,Uniform{Float64},<:Vector}

    ds2 = Fill(Uniform(0.0, first(ubound)), N)
    d_product2 = @inferred(product_distribution(ds2))
    @test d_product2 isa Distributions.VectorOfUnivariateDistribution{Continuous,Uniform{Float64},<:Fill}

    # Check that methods for `VectorOfUnivariateDistribution` are consistent.
    for (ds, d_product) in ((ds1, d_product1), (ds2, d_product2))
        @test length(d_product) == length(ds)
        @test eltype(d_product) === eltype(ds[1])
        @test @inferred(mean(d_product)) == mean.(ds)
        @test @inferred(var(d_product)) == var.(ds)
        @test @inferred(cov(d_product)) == Diagonal(var.(ds))
        @test @inferred(entropy(d_product)) == sum(entropy.(ds))
        @test insupport(d_product, zeros(N))
        @test insupport(d_product, maximum.(ds))
        @test !insupport(d_product, maximum.(ds) .+ 1)
        @test !insupport(d_product, zeros(N + 1))

        x = @inferred(rand(d_product))
        @test x isa typeof(rand.(collect(ds)))
        @test length(x) == length(d_product)
        @test insupport(d_product, x)
        @test @inferred(logpdf(d_product, x)) ≈ sum(logpdf.(ds, x))
        # ensure that samples are different, in particular if `Fill` is used
        @test length(unique(x)) == N
    end
end

@testset "Testing discrete non-parametric VectorOfUnivariateDistribution" begin
    Random.seed!(123456)
    N = 11

    for a in ([0, 1], [-0.5, 0.5])
        # Construct independent distributions and `ProductDistribution` from these.
        ds1 = DiscreteNonParametric.(fill(a, N), Ref([0.5, 0.5]))
        d_product1 = @inferred(product_distribution(ds1))
        @test d_product1 isa Distributions.VectorOfUnivariateDistribution{Discrete,<:DiscreteNonParametric,<:Vector}

        ds2 = Fill(DiscreteNonParametric(a, [0.5, 0.5]), N)
        d_product2 = @inferred(product_distribution(ds2))
        @test d_product2 isa Distributions.VectorOfUnivariateDistribution{Discrete,<:DiscreteNonParametric,<:Fill}

        # Check that methods for `VectorOfUnivariateDistribution` are consistent.
        for (ds, d_product) in ((ds1, d_product1), (ds2, d_product2))
            @test length(d_product) == length(ds)
            @test eltype(d_product) === eltype(ds[1])
            @test @inferred(mean(d_product)) == mean.(ds)
            @show typeof(d_product)
            @show @inferred(map(probs, d_product.v))
            @test @inferred(var(d_product)) == var.(ds)
            @test @inferred(cov(d_product)) == Diagonal(var.(ds))
            @test @inferred(entropy(d_product)) == sum(entropy.(ds))
            @test insupport(d_product, fill(a[2], N))
            @test !insupport(d_product, fill(a[2] + 1, N))
            @test !insupport(d_product, fill(a[2], N + 1))

            x = @inferred(rand(d_product))
            @test x isa typeof(rand.(collect(ds)))
            @test length(x) == length(d_product)
            @test insupport(d_product, x)
            @test @inferred(logpdf(d_product, x)) ≈ sum(logpdf.(ds, x))
            # ensure that samples are different, in particular if `Fill` is used
            @test length(unique(x)) == 2
        end
    end
end

@testset "Testing generic MatrixOfUnivariateDistribution" begin
    Random.seed!(123456)
    M, N = 11, 16

    # Construct independent distributions and `ProductDistribution` from these.
    ubound = rand(M, N)

    ds1 = Uniform.(0.0, ubound)
    d_product1 = @inferred(product_distribution(ds1))
    @test d_product1 isa Distributions.MatrixOfUnivariateDistribution{Continuous,Uniform{Float64},<:Matrix}

    ds2 = Fill(Uniform(0.0, first(ubound)), M, N)
    d_product2 = @inferred(product_distribution(ds2))
    @test d_product2 isa Distributions.MatrixOfUnivariateDistribution{Continuous,Uniform{Float64},<:Fill{Uniform{Float64},2}}

    # Check that methods for `MatrixOfUnivariateDistribution` are consistent.
    for (ds, d_product) in ((ds1, d_product1), (ds2, d_product2))
        @test size(d_product) == size(ds)
        @test eltype(d_product) === eltype(ds[1])
        @test @inferred(mean(d_product)) == mean.(ds)
        @test @inferred(var(d_product)) == var.(ds)
        @test @inferred(cov(d_product)) == Diagonal(vec(var.(ds)))
        @test @inferred(cov(d_product, Val(false))) == reshape(Diagonal(vec(var.(ds))), M, N, M, N)

        x = @inferred(rand(d_product))
        @test size(x) == size(d_product)
        @test x isa typeof(rand.(collect(ds)))
        @test @inferred(logpdf(d_product, x)) ≈ sum(logpdf.(ds, x))
        # ensure that samples are different, in particular if `Fill` is used
        @test length(unique(x)) == length(d_product)
    end
end

@testset "Testing generic VectorOfMultivariateDistribution" begin
    Random.seed!(123456)
    M, N = 3, 11

    # Construct independent distributions and `ProductDistribution` from these.
    alphas = [normalize!(rand(M), 1) for _ in 1:N]

    ds1 = Dirichlet.(alphas)
    d_product1 = @inferred(product_distribution(ds1))
    @test d_product1 isa Distributions.VectorOfMultivariateDistribution{Continuous,<:Dirichlet{Float64},<:Vector}

    ds2 = Fill(Dirichlet(first(alphas)), N)
    d_product2 = @inferred(product_distribution(ds2))
    @test d_product2 isa Distributions.VectorOfMultivariateDistribution{Continuous,<:Dirichlet{Float64},<:Fill{<:Dirichlet{Float64},1}}

    # Check that methods for `VectorOfMultivariateDistribution` are consistent.
    for (ds, d_product) in ((ds1, d_product1), (ds2, d_product2))
        @test size(d_product) == (length(ds[1]), length(ds))
        @test eltype(d_product) === eltype(ds[1])
        @test @inferred(mean(d_product)) == reduce(hcat, mean.(ds))
        @test @inferred(var(d_product)) == reduce(hcat, var.(ds))
        @test @inferred(cov(d_product)) == Diagonal(reduce(vcat, var.(ds)))
        @test @inferred(cov(d_product, Val(false))) == reshape(Diagonal(reduce(vcat, var.(ds))), M, N, M, N)

        x = @inferred(rand(d_product))
        @test size(x) == size(d_product)
        @test x isa typeof(mapreduce(rand, hcat, ds))
        @test @inferred(logpdf(d_product, x)) ≈ sum(logpdf(d, x[:, i]) for (i, d) in enumerate(ds))
        # ensure that samples are different, in particular if `Fill` is used
        @test length(unique(x)) == length(d_product)
    end
end
