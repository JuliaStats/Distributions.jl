using Distributions
using Base.Test

for d in (Truncated(Normal(0, 1), -1, 1),
          Truncated(Normal(3, 10), 7, 8),
          Truncated(Normal(-5, 1), -Inf, -10))
    @test all(insupport(d, rand(d, 1000)))
end

d = Truncated(Normal(0, 1), -0.1, +0.1)

@test pdf(d, 0.0) > pdf(Normal(0, 1), 0.0)
@test pdf(d, -1.0) == 0.0
@test pdf(d, +1.0) == 0.0

@test logpdf(d, 0.0) > logpdf(Normal(0, 1), 0.0)
@test isinf(logpdf(d, -1.0))
@test isinf(logpdf(d, +1.0))

@test cdf(d, -1.0) == 0.0
@test cdf(d, -0.09) < cdf(Normal(0, 1), -0.09)
@test cdf(d, 0.0) == 0.5
@test cdf(d, +0.09) > cdf(Normal(0, 1), +0.09)
@test cdf(d, +1.0) == 1.0

@test quantile(d, 0.01) > -0.1
@test abs(quantile(d, 0.5) - 0.0) < 1e-8
@test quantile(d, 0.99) < +0.1

@test abs(cdf(d, quantile(d, 0.01)) - 0.01) < 1e-8
@test abs(cdf(d, quantile(d, 0.50)) - 0.50) < 1e-8
@test abs(cdf(d, quantile(d, 0.99)) - 0.99) < 1e-8


