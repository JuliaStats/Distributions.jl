using Test

using Distributions

@test convert(Binomial, Bernoulli(0.75)) == Binomial(1, 0.75)

@test convert(Gamma, Exponential(3.0)) == Gamma(1.0, 3.0)
@test convert(Gamma, Erlang(5, 2.0)) == Gamma(5.0, 2.0)
