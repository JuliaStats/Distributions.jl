using Distributions, Test, Random, LinearAlgebra
using Distributions: Product

@testset "Testing Product distributions" begin
let
    rng, D = MersenneTwister(123456), 11

    # Construct independent distributions and `Product` distribution from these.
    μ = randn(rng, D)
    ds = Normal.(μ, 1.0)
    x = rand.(Ref(rng), ds)
    d_product = Product(ds)

    # Check that methods for `Product` are consistent.
    @test length(d_product) == length(ds)
    @test logpdf(d_product, x) ≈ sum(logpdf.(ds, x))
    @test mean(d_product) == mean.(ds)
    @test var(d_product) == var.(ds)
    @test cov(d_product) == Diagonal(var.(ds))
    @test entropy(d_product) == sum(entropy.(ds))
end
end
