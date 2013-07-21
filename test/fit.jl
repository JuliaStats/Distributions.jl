using Distributions
using Base.Test

N = 10^5

d = fit(DiscreteUniform, rand(DiscreteUniform(10, 15), N))
@test isa(d, DiscreteUniform)
@test min(d) == 10
@test max(d) == 15

d = fit(Bernoulli, rand(Bernoulli(0.7), N))
@test isa(d, Bernoulli)
@test_approx_eq_eps mean(d) 0.7 0.01

d = fit(Beta, rand(Beta(1.3, 3.7), N))
@test isa(d, Beta)
@test_approx_eq_eps d.alpha 1.3 0.1
@test_approx_eq_eps d.beta  3.7 0.1

d = fit(Binomial, 100, rand(Binomial(100, 0.3), N))
@test isa(d, Binomial)
@test d.size == 100
@test_approx_eq_eps d.prob 0.3 0.01

d = fit(Exponential, rand(Exponential(0.5), N))
@test isa(d, Exponential)
@test_approx_eq_eps mean(d) 0.5 0.01

x = rand(Normal(11.3, 5.7), N)
d = fit(Normal, x)
@test isa(d, Normal)
@test_approx_eq_eps mean(d) 11.3 0.1
@test_approx_eq_eps std(d) 5.7 0.1

d = fit_mle(Normal, suffstats(Normal, x))
@test isa(d, Normal)
@test_approx_eq_eps mean(d) 11.3 0.1
@test_approx_eq_eps std(d) 5.7 0.1

d = fit(Uniform, rand(Uniform(1.2, 10.3), N))
@test isa(d, Uniform)
@test 1.2 <= min(d) <= max(d) <= 10.3
@test_approx_eq_eps min(d) 1.2 0.02
@test_approx_eq_eps max(d) 10.3 0.02

# 

# # TODO: Reable when polygamma gets merged
# # fit(Gamma, rand(Gamma(7.9, 3.1), N))

# fit(Geometric, rand(Geometric(0.1), N))

# fit(Laplace, rand(Laplace(10.0, 3.0), N))

# 

# fit(Poisson, rand(Poisson(19.0), N))

# 
