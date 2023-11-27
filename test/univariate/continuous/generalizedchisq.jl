using Distributions
using Test
using Random

@testset "Generalized chi-squared" begin
    rng = Random.MersenneTwister(0)
    Computations = Distributions.GChisqComputations
    quadgk = Distributions.quadgk
    # General case
    gx2 = GeneralizedChisq([1,-1], [1,2], [1.5, 1.5], 10, 1)
    @test eltype(gx2) == typeof(gx2.μ) <: AbstractFloat
    @test minimum(gx2) == -Inf
    @test maximum(gx2) == Inf
    @test insupport(gx2, rand(gx2))
    @test let t = randn()
        cf(gx2, t) ≈ Computations.cf_inherit(gx2, t)
    end
    @test cdf(gx2, 15) - cdf(gx2, 5) ≈ first(quadgk(x->pdf(gx2, x), 5, 15))
    @test pdf(gx2, 15) - pdf(gx2, 5) ≈ first(quadgk(x->Computations.daviesdpdf(gx2, x), 5, 15))
    @testset for p in 0:0.05:1
        q = quantile(gx2, p)
        @test cdf(gx2, q) ≈ p
    end
    # Other types
    gx2 = GeneralizedChisq([1,1], [1,1], BigFloat[0,1], 10, 1)
    @test eltype(gx2) == typeof(gx2.μ) <: BigFloat
    # [Don't test with BigFloat]
    gx2 = GeneralizedChisq([1,1], [1,1], [0,1], 10, 1f0)
    @test eltype(gx2) == typeof(gx2.μ) <: Float32
    @test insupport(gx2, rand(gx2))
    # Special cases:
    # σ == 0, positive weights
    gx2 = GeneralizedChisq([1,1], [1,2], [0,1], 10, 0)
    @test all(≥(10), rand(gx2, 10))
    @test minimum(gx2) == 10
    @test let t = randn()
        cf(gx2, t) ≈ Computations.cf_inherit(gx2, t)
    end
    @test cdf(gx2, 15) - cdf(gx2, 5) ≈ first(quadgk(x->pdf(gx2, x), 5, 15)) ≈ first(quadgk(x->pdf(gx2, x), 10, 15))
    @testset for p in 0:0.05:1
        q = quantile(gx2, p)
        @test cdf(gx2, q) ≈ p
    end
    # σ == 0, negative weights
    gx2 = GeneralizedChisq([-1,-1], [1,2], [0,1], 10, 0)
    @test all(≤(10), rand(gx2, 10))
    @test maximum(gx2) == 10
    @test let t = randn()
        cf(gx2, t) ≈ Computations.cf_inherit(gx2, t)
    end
    @test cdf(gx2, 15) - cdf(gx2, 5) ≈ first(quadgk(x->pdf(gx2, x), 5, 15)) ≈ first(quadgk(x->pdf(gx2, x), 5, 10))
    @testset for p in 0:0.05:1
        q = quantile(gx2, p)
        @test cdf(gx2, q) ≈ p
    end
    # All zero weights
    gx2 = GeneralizedChisq([0,0], [1,2], [0,1], 10, 1)
    gx2b = GeneralizedChisq(Int[], Int[], Int[], 10, 1)
    normalequiv = Normal(gx2.μ, gx2.σ)
    @test let t = randn()
        cf(gx2, t) ≈ cf(gx2b, t) ≈ Computations.cf_inherit(gx2, t) ≈ cf(normalequiv, t)
    end
    let x = 10 + randn()
        @test cdf(gx2, x) ≈ cdf(gx2b, x) ≈ Computations.daviescdf(gx2, x) ≈ cdf(normalequiv, x)
        @test pdf(gx2, x) ≈ pdf(gx2b, x) ≈ Computations.daviespdf(gx2, x) ≈ pdf(normalequiv, x)
    end
    @testset for p in 0:0.05:1
        q = quantile(gx2, p)
        @test q == quantile(gx2, p)
        @test cdf(gx2, q) ≈ cdf(gx2b, q) ≈ p
    end
    # delta at μ
    gx2 = GeneralizedChisq([0,0], [1,2], [0,1], 10, 0)
    gx2b = GeneralizedChisq(Int[], Int[], Int[], 10, 0)
    @test let t = randn()
        cf(gx2, t) ≈ cf(gx2b, t) ≈ Computations.cf_inherit(gx2, t)
    end
    let x = 9
        @test cdf(gx2, x) == cdf(gx2b, x) == 0
        @test pdf(gx2, x) == pdf(gx2b, x) == 0
    end
    let x = 10
        @test cdf(gx2, x) == cdf(gx2b, x) == 1
        @test pdf(gx2, x) == pdf(gx2b, x) == Inf
    end
    let x = 11
        @test cdf(gx2, x) == cdf(gx2b, x) == 1
        @test pdf(gx2, x) == pdf(gx2b, x) == 0
    end
    @testset for p in 0:0.05:1
        q = quantile(gx2, p)
        @test q == quantile(gx2, p) == 10
    end
end