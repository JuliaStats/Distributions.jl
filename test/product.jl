using Distributions, Test, Random, LinearAlgebra
using Distributions: Product

@testset "Testing normal product distributions" begin
    Random.seed!(123456)
    N = 11
    # Construct independent distributions and `Product` distribution from these.
    μ = randn(N)
    ds = Normal.(μ, 1.0)
    x = rand.(ds)
    d_product = product_distribution(ds)
    @test d_product isa MvNormal
    # Check that methods for `Product` are consistent.
    @test length(d_product) == length(ds)
    @test eltype(d_product) === eltype(ds[1])
    @test logpdf(d_product, x) ≈ sum(logpdf.(ds, x))
    @test mean(d_product) == mean.(ds)
    @test var(d_product) == var.(ds)
    @test cov(d_product) == Diagonal(var.(ds))
    @test entropy(d_product) ≈ sum(entropy.(ds))

    y = rand(d_product)
    @test y isa typeof(x)
    @test length(y) == N
end

@testset "Testing generic product distributions" begin
    Random.seed!(123456)
    N = 11
    # Construct independent distributions and `Product` distribution from these.
    ubound = rand(N)
    ds = Uniform.(-ubound, ubound)
    x = rand.(ds)
    d_product = product_distribution(ds)
    @test d_product isa Product
    # Check that methods for `Product` are consistent.
    @test length(d_product) == length(ds)
    @test eltype(d_product) === eltype(ds[1])
    @test logpdf(d_product, x) ≈ sum(logpdf.(ds, x))
    @test mean(d_product) == mean.(ds)
    @test var(d_product) == var.(ds)
    @test cov(d_product) == Diagonal(var.(ds))
    @test entropy(d_product) == sum(entropy.(ds))
    @test insupport(d_product, ubound) == true
    @test insupport(d_product, ubound .+ 1) == false
    @test minimum(d_product) == -ubound
    @test maximum(d_product) == ubound
    @test extrema(d_product) == (-ubound, ubound)
    @test isless(extrema(d_product)...)

    y = rand(d_product)
    @test y isa typeof(x)
    @test length(y) == N
end

@testset "Testing discrete non-parametric product distribution" begin
    Random.seed!(123456)
    N = 11

    for a in ([0, 1], [-0.5, 0.5])
        # Construct independent distributions and `Product` distribution from these.
        support = fill(a, N)
        ds = DiscreteNonParametric.(support, Ref([0.5, 0.5]))
        x = rand.(ds)
        d_product = product_distribution(ds)
        @test d_product isa Product
        # Check that methods for `Product` are consistent.
        @test length(d_product) == length(ds)
        @test eltype(d_product) === eltype(ds[1])
        @test logpdf(d_product, x) ≈ sum(logpdf.(ds, x))
        @test mean(d_product) == mean.(ds)
        @test var(d_product) == var.(ds)
        @test cov(d_product) == Diagonal(var.(ds))
        @test entropy(d_product) == sum(entropy.(ds))
        @test insupport(d_product, fill(a[2], N)) == true
        @test insupport(d_product, fill(a[2] + 1, N)) == false

        y = rand(d_product)
        @test y isa typeof(x)
        @test length(y) == N
    end
end
