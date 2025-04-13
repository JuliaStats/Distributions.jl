using Distributions
using StableRNGs
using Test

d = Semicircle(2.0)

@test params(d) == (2.0,)

@test minimum(d) == -2.0
@test maximum(d) == +2.0
@test extrema(d) == (-2.0, 2.0)

@test mean(d)     ==  .0
@test var(d)      == 1.0
@test skewness(d) ==  .0
@test median(d)   ==  .0
@test mode(d)     ==  .0
@test entropy(d)  == 1.33787706640934544458

@test pdf(d, -5) == .0
@test pdf(d, -2) == .0
@test pdf(d,  0) == .31830988618379067154
@test pdf(d, +2) == .0
@test pdf(d, +5) == .0

@test logpdf(d, -5) == -Inf
@test logpdf(d, -2) == -Inf
@test logpdf(d,  0) ≈  log(.31830988618379067154)
@test logpdf(d, +2) == -Inf
@test logpdf(d, +5) == -Inf

@test cdf(d, -5) ==  .0
@test cdf(d, -2) ==  .0
@test cdf(d,  0) ==  .5
@test cdf(d, +2) == 1.0
@test cdf(d, +5) == 1.0

@test quantile(d,  .0) == -2.0
@test quantile(d,  .5) ==   .0
@test quantile(d, 1.0) == +2.0

rng = StableRNG(123)
@testset for r in rand(rng, Uniform(0,10), 5)
    N = 10^4
    semi = Semicircle(r)
    sample = rand(rng, semi, N)
    mi, ma = extrema(sample)
    @test -r <= mi < ma <= r

    # test order statistic of sample min is sane
    d_min = Beta(1, N)
    lo = quantile(d_min, 0.01)
    hi = quantile(d_min, 0.99)
    @test lo < cdf(semi, mi) < hi

    # test order statistic of sample max is sane
    d_max = Beta(N, 1)
    lo = quantile(d_max, 0.01)
    hi = quantile(d_max, 0.99)
    @test lo < cdf(semi, ma) < hi

    # central limit theorem
    dmean = Normal(mean(semi), std(semi)/√(N))
    @test quantile(dmean, 0.01) < mean(sample) < quantile(dmean, 0.99)

    pvalue = pvalue_kolmogorovsmirnoff(sample, semi)
    @test pvalue > 1e-2
end
