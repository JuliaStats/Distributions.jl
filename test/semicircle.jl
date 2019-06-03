using Distributions
using  Test

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
