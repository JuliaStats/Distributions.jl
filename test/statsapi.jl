using Distributions
using StatsAPI: pvalue

using Test

@testset "pvalue" begin
    # For two discrete and two continuous distribution
    for dist in (Binomial(10, 0.3), Poisson(0.3), Normal(1.4, 2.1), Gamma(1.9, 0.8))
        # Draw sample
        x = rand(dist)

        # Draw 10^6 additional samples
        ys = rand(dist, 1_000_000)

        # Check that empirical frequencies match pvalues of left/right tail approximately
        @test pvalue(dist, x; tail=:left) ≈ mean(≤(x), ys) rtol=5e-3
        @test pvalue(dist, x; tail=:right) ≈ mean(≥(x), ys) rtol=5e-3

        # Check consistency of pvalues of both tails
        @test pvalue(dist, x; tail=:both) ==
            min(1, 2 * min(pvalue(dist, x; tail=:left), pvalue(dist, x; tail=:right)))

        # Incorrect value for keyword argument
        @test_throws ArgumentError("`tail=:l` is invalid") pvalue(dist, x; tail=:l)
    end
end
