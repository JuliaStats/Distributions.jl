using Distributions
import Distributions.quadgk
using Tests
using Random

@testset "Generalized chi-squared" begin
    rng = Random.MersenneTwister(0)
    # check types of corner cases
    gx2 = GeneralizedChisq([1,1], [1,1], [0,1], 10, 1)
    @test typeof(gx2.μ) <: AbstractFloat
    gx2 = GeneralizedChisq([1,1], [1,1], BigFloat[0,1], 10, 1)
    @test typeof(gx2.μ) <: BigFloat
    gx2 = GeneralizedChisq(Int[], Int[], Int[], 0, 1.0)
    @test typeof(gx2.μ) == typeof(1.0)
    # test sampler
    # test maximum-minimum
    # test cf, cdf, pdf
    gx2 = GeneralizedChisq([1,-1], [1,2], [1.5, 1.5], 10, 1)
    @test let t = 10 + randn()
        cf(gx2, t) ≈ GChisqComputations.cf_inherit(gx2, t)
    end
    @test cdf(gx2, 15) - cdf(gx2, 5) ≈ first(quadgk(x->pdf(gx2, x), 5, 15))
    @test pdf(gx2, 15) - pdf(gx2, 5) ≈ first(quadgk(x->GChisqComputations.daviesdpdf(gx2, x), 5, 15))
    # test search of convergence for starting point of quantile
    for p in 0.1:0.1:0.9, x in 0:15
        println(x, ", ", p, ", ", GChisqComputations.newtonconvergence(gx2, p, x))
    end
    for p in 0.0:0.05:1.0
        println(p, ", ", GChisqComputations.newtonconvergence(gx2, p, gx2.μ))
    end
    # using BenchmarkTools
    for p in 0.05:0.05:0.9
        qn = quantile(gx2, p)
        tn = @elapsed quantile(gx2, p)
        qb = quantile_b(gx2, p)
        tb = @elapsed quantile_b(gx2, p)
        println(p, ", ", qn, ", ", tn, ", ", cdf(gx2, qn))
        println(p, ", ", qb, ", ", tb, ", ", cdf(gx2, qb))
        println()
    end
end