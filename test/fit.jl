using Distributions
using Base.Test

N = 10^5

d = fit(Bernoulli, rand(Bernoulli(0.7), N))
@test isa(d, Bernoulli)
@test_approx_eq_eps mean(d) 0.7 0.01

d = fit(Beta, rand(Beta(1.3, 3.7), N))
@test isa(d, Beta)
@test_approx_eq_eps d.alpha 1.3 0.01
@test_approx_eq_eps d.beta  3.7 0.01

d = fit(Binomial, 100, rand(Binomial(100, 0.3), N))
@test isa(d, Binomial)
@test d.size == 100
@test_approx_eq_eps d.prob 0.3 0.01

d = fit(Exponential, rand(Exponential(0.5), N))
@test isa(d, Exponential)
@test_approx_eq_eps mean(d) 0.5 0.01

# fit(DiscreteUniform, rand(DiscreteUniform(300_000, 700_000), N))

# 

# # TODO: Reable when polygamma gets merged
# # fit(Gamma, rand(Gamma(7.9, 3.1), N))

# fit(Geometric, rand(Geometric(0.1), N))

# fit(Laplace, rand(Laplace(10.0, 3.0), N))

# fit(Normal, rand(Normal(11.3, 5.7), N))

# fit(Poisson, rand(Poisson(19.0), N))

# fit(Uniform, rand(Uniform(1.1, 98.3), N))
