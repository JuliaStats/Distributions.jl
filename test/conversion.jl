using Test
using Distributions

@test convert(Binomial, Bernoulli(0.75)) == Binomial(1, 0.75)

@test convert(Gamma, Exponential(3.0)) == Gamma(1.0, 3.0)
@test convert(Gamma, Erlang(5, 2.0)) == Gamma(5.0, 2.0)


@test convert(Stable, Normal(42., 2.)) ≈ Stable(2., 0., √2, 42.)
@test convert(Stable, Cauchy(7., 13.)) == Stable(1., 0., 13., 7.)
@test convert(Stable, Levy(3., 5.)) == Stable(0.5, 1., 5., 3.)
