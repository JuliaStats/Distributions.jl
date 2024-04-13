using Distributions
using FillArrays

using LinearAlgebra
using Random
using Test

using Distributions: Product

# TODO: remove when `Product` is removed
@testset "Deprecated `Product` distribution" begin
@testset "Testing normal product distributions" begin
    Random.seed!(123456)
    N = 11
    # Construct independent distributions and `Product` distribution from these.
    μ = randn(N)
    ds = Normal.(μ, 1.0)
    x = rand.(ds)
    d_product = @test_deprecated(Product(ds))
    @test d_product isa Product
    # Check that methods for `Product` are consistent.
    @test length(d_product) == length(ds)
    @test eltype(d_product) === eltype(ds[1])
    @test @inferred(logpdf(d_product, x)) ≈ sum(logpdf.(ds, x))
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
    @test @inferred(logpdf(d_product, x)) ≈ sum(logpdf.(ds, x))
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
        ds = [DiscreteNonParametric(copy(a), [0.5, 0.5]) for _ in 1:N]
        x = rand.(ds)
        d_product = product_distribution(ds)
        @test d_product isa Product
        # Check that methods for `Product` are consistent.
        @test length(d_product) == length(ds)
        @test eltype(d_product) === eltype(ds[1])
        @test @inferred(logpdf(d_product, x)) ≈ sum(logpdf.(ds, x))
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

@testset "Testing iid product distributions" begin
    Random.seed!(123456)
    N = 11
    d = @test_deprecated(Product(Fill(Laplace(0.0, 2.3), N)))
    @test N == length(unique(rand(d)));
    @test mean(d) === Fill(0.0, N)
    @test cov(d) === Diagonal(Fill(var(Laplace(0.0, 2.3)), N))
end

@testset "Empty vector of distributions (#1619)" begin
    d = @inferred(product_distribution(typeof(Beta(1, 1))[]))
    @test d isa Product
    @test iszero(@inferred(logpdf(d, Float64[])))
    @test_throws DimensionMismatch logpdf(d, rand(1))
    @test_throws DimensionMismatch logpdf(d, rand(3))
end
end