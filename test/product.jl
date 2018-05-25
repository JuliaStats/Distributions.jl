using Distributions
using Compat.Test

using Distributions: Product
let
    rng, D = MersenneTwister(123456), 11

    # Construct independent distributions and `Product` distribution from these.
    μ = randn(rng, D)
    ds = Normal.(μ, 1.0)
    x = rand.(rng, ds)
    d_product = Product(ds)

    # Check that methods for `Product` are consistent.
    @test length(d_product) == length(ds)
    @test logpdf(d_product, x) ≈ sum(logpdf.(ds, x))
end
