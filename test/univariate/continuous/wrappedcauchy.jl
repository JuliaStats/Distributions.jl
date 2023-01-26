using Distributions
using Random: MersenneTwister
using Test

d = WrappedCauchy(0.6)

@test mean(d)     ==  .0
@test var(d)      == 0.4
@test skewness(d) ==  .0
@test median(d)   ==  .0
@test mode(d)     ==  .0
@test entropy(d)  == 1.391589963780926

@test pdf(d, -5) == .0
@test pdf(d, 2π) == .0
@test pdf(d,  0) == 0.6366197723675816

@test logpdf(d, -5) == -Inf
@test logpdf(d,  0) ≈  log(0.6366197723675816)
@test logpdf(d, +5) == -Inf

@test cdf(d, -5) ==  .0
@test cdf(d,  0) ==  .5
@test cdf(d, +5) == 1.0
