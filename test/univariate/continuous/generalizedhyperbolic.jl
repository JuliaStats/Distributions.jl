@testset "Generalized hyperbolic" begin
    Hyp(z, p=0, μ=0, σ=1, λ=1) = GeneralizedHyperbolic(Val(:locscale), z, p, μ, σ, λ)
    # Empirical characteristic function
    cf_empirical(samples::AbstractVector{<:Real}, t::Real) = mean(x->exp(1im * t * x), samples)

    N = 10^6 # number of samples
    distributions = [
        # No skewness, location, scale
        Hyp(3/10), Hyp(3), Hyp(10),
        # Add skewness
        Hyp(1/10, -5), Hyp(3, -1), Hyp(8, 1), Hyp(20, 5), # last one breaks `test_samples`
        # Add location & scale
        Hyp(1, -2, -1, 5), Hyp(1, -2, 1, 5), Hyp(6, 1, -5, 2),
        # Different λ
        Hyp(3/10, 0,0,1, -5), Hyp(3, 0,0,1, -1), Hyp(10, 0,0,1, 4),
        Hyp(3, -1, 0,1, -1), Hyp(8, 1, 0,1, 5),
        Hyp(1, -2, -1, 5, -1/2), Hyp(1, -2, -1, 5, 1/2), Hyp(6, 1, -5, 2, 8),
    ]
    modes = [
        0, 0, 0,
        -5, -1, 1, 5,
        -11, -9, -3,
        0, 0, 0,
        -0.560464077632561342794140059659, 1.60212984628092067584739859051,
        -4.39143813989001345842141045503, -7.49187863276682628498920382058, 0.327863171219295387089685071504
    ]
    for (i, (d, mode_true)) in enumerate(zip(distributions, modes))
        println("\ttesting $d")

        @test isapprox(mode(d), mode_true, atol=1e-6)

        samples = rand(d, N)
        @test abs(mean(d) - mean(samples)) < 0.02
		@test abs(std(d) - std(samples)) < 0.05
        
        # Empirical CF should be close to theoretical CF
        @test maximum(t->abs(cf(d, t) - cf_empirical(samples, t)), range(-100, 100, 100)) < 0.005

        @test length(test_samples(d, N)) > 0 broken=(i == 7)
    end
end
