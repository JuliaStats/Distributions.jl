using Distributions, Test, Random, LinearAlgebra
using Distributions: Product

@testset "Testing normal product distributions" begin
let
    rng = MersenneTwister(123456)
    N = 11
    # Construct independent distributions and `Product` distribution from these.
    μ = randn(rng, N)
    ds = Normal.(μ, 1.0)
    x = rand.(Ref(rng), ds)
    d_product = product_distribution(ds)
    @test d_product isa MvNormal
    # Check that methods for `Product` are consistent.
    @test length(d_product) == length(ds)
    @test logpdf(d_product, x) ≈ sum(logpdf.(ds, x))
    @test mean(d_product) == mean.(ds)
    @test var(d_product) == var.(ds)
    @test cov(d_product) == Diagonal(var.(ds))
    @test entropy(d_product) ≈ sum(entropy.(ds))
end
end

@testset "Testing generic product distributions" begin
let
    rng = MersenneTwister(123456)
    N = 11
    # Construct independent distributions and `Product` distribution from these.
    ubound = rand(rng, N)
    ds = Uniform.(0.0, ubound)
    x = rand.(Ref(rng), ds)
    d_product = product_distribution(ds)
    @test d_product isa Product
    # Check that methods for `Product` are consistent.
    @test length(d_product) == length(ds)
    @test logpdf(d_product, x) ≈ sum(logpdf.(ds, x))
    @test mean(d_product) == mean.(ds)
    @test var(d_product) == var.(ds)
    @test cov(d_product) == Diagonal(var.(ds))
    @test entropy(d_product) == sum(entropy.(ds))
end
end
