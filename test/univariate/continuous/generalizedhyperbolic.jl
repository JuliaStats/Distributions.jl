@testset "Generalized hyperbolic" begin
    Hyp(z, p=0, μ=0, σ=1, λ=1) = GeneralizedHyperbolic(Val(:locscale), z, p, μ, σ, λ)
    # Empirical characteristic function
    cf_empirical(samples::AbstractVector{<:Real}, t::Real) = mean(x->exp(1im * t * x), samples)

    N = 10^6 # number of samples
    distributions = [
        # No skewness, location, scale
        Hyp(0.3), Hyp(3), Hyp(10),
        # Add skewness
        Hyp(0.1, -5), Hyp(3, -1), Hyp(8, 1), Hyp(20, 5),
        # Add location & scale
        Hyp(1, -2, -1, 5), Hyp(1, -2, 1, 5), Hyp(6, 1, -5, 2),
        # Different λ
        Hyp(8, 1, -2, 3, -1/2), Hyp(1, -8, 2, 0.4, 1/2), Hyp(1, -8, 2, 0.4, 5),
    ]
    for d in distributions
        println("\ttesting $d")

        samples = rand(d, N)
        @test abs(mean(d) - mean(samples)) < 0.01
		@test abs(std(d) - std(samples)) < 0.01
        # Empirical CF should be close to theoretical CF
        @test maximum(t->abs(cf(d, t) - cf_empirical(samples, t)), range(-100, 100, 100)) < 0.005

        test_samples(d, N)
    end
end
