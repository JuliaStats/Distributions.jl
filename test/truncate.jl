d = TruncatedNormal(Normal(0, 1), -0.1, +0.1)

@assert pdf(d, 0.0) > pdf(Normal(0, 1), 0.0)
@assert pdf(d, -1.0) == 0.0
@assert pdf(d, +1.0) == 0.0

@assert logpdf(d, 0.0) > logpdf(Normal(0, 1), 0.0)
@assert isinf(logpdf(d, -1.0))
@assert isinf(logpdf(d, +1.0))

@assert cdf(d, -1.0) == 0.0
@assert cdf(d, -0.09) < cdf(Normal(0, 1), -0.09)
@assert cdf(d, 0.0) == 0.5
@assert cdf(d, +0.09) > cdf(Normal(0, 1), +0.09)
@assert cdf(d, +1.0) == 1.0

@assert quantile(d, 0.01) > -0.1
@assert abs(quantile(d, 0.5) - 0.0) < 1e-8
@assert quantile(d, 0.99) < +0.1

@assert abs(cdf(d, quantile(d, 0.01)) - 0.01) < 1e-8
@assert abs(cdf(d, quantile(d, 0.50)) - 0.50) < 1e-8
@assert abs(cdf(d, quantile(d, 0.99)) - 0.99) < 1e-8
